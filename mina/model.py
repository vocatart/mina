import math
from typing import NamedTuple

import lightning
import numpy as np
import torch
from muon import SingleDeviceMuonWithAuxAdam
from torch import nn, Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mina.acoustic import ConvAcousticEncoder

import matplotlib
import matplotlib.pyplot as plt

from mina.phoneme import PhonemeClassifier
from mina.positional_encoding import PositionalEncodingType
from mina.temporal import TemporalContextEncoder

EPSILON = 1e-8

class MINA(lightning.LightningModule):
    def __init__(self, d_mel: int, d_l: int, d_h: int, conv_layers: int,
                 num_heads: int, tf_layers: int, tf_dim_ff: int, dropout_conv: float,
                 dropout_tf: float, kernel_size: int, max_len: int, sr: int, phoneme_dropout: float,
                 hop_length: int, muon_lr: float, adam_lr: float, pos_weight: float,
                 boundary_threshold: float, pe_type: PositionalEncodingType, vocab_size: int,
                 weight_decay: float, warmup_steps: int, sch_frequency: int, do_compile: bool,
                 phoneme_map: dict[int, str], loss_weights: tuple[float, float, float],
                 hit_tolerance: float):
        super().__init__()
        self.save_hyperparameters()

        self.acoustic = ConvAcousticEncoder(d_mel, d_l, d_h, conv_layers, kernel_size, dropout_conv)
        self.temporal = TemporalContextEncoder(d_h, num_heads, tf_layers, tf_dim_ff, dropout_tf, max_len, pe_type)
        self.boundary_classifier = nn.Linear(d_h, 1)
        self.phoneme_classifier = PhonemeClassifier(d_h, vocab_size, phoneme_dropout)

        self.example_input_array = torch.randn(1, max_len, d_mel)


    def forward(self, x: torch.Tensor, padding_mask=None, gt_boundaries=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.acoustic(x)
        x = self.temporal(x, padding_mask=padding_mask)

        boundary_logits = self.boundary_classifier(x).squeeze(-1)
        phoneme_logits, segment_logits = self.phoneme_classifier(x, padding_mask=padding_mask, gt_boundaries=gt_boundaries)

        return boundary_logits, phoneme_logits, segment_logits

    def on_train_start(self):
        if self.hparams.do_compile:
            self.acoustic.compile(dynamic=True)
            self.temporal.compile(dynamic=True)
            self.boundary_classifier.compile(dynamic=True)
            self.phoneme_classifier.compile(dynamic=True)

    @staticmethod
    def _make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        Creates a boolean padding mask using original lengths from batch

        Args:
            lengths (torch.Tensor): Tensor of lengths
            max_len (int): Max sequence length

        Returns:
            Boolean mask tensor
        """

        idx = torch.arange(max_len, device=lengths.device)
        return idx.unsqueeze(0) >= lengths.unsqueeze(1)

    def compute_loss(self,
                     boundary_logits: torch.Tensor,
                     phoneme_logits: torch.Tensor,
                     segment_logits: torch.Tensor,
                     gt_boundaries: torch.Tensor,
                     gt_phonemes: torch.Tensor,
                     gt_segments: torch.Tensor,
                     frame_mask: torch.Tensor,
                     seg_mask: torch.Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Binary cross entropy loss with mask and weighting

        Args:
            boundary_logits (torch.Tensor): Tensor of boundary logits
            phoneme_logits (torch.Tensor): Tensor of phoneme logits
            segment_logits (torch.Tensor): Tensor of segment logits
            gt_boundaries (torch.Tensor): Tensor of gt boundaries
            gt_phonemes (torch.Tensor): Tensor of gt phonemes
            gt_segments (torch.Tensor): Tensor of gt segments
            frame_mask (torch.Tensor): Boolean mask of frame tensor
            seg_mask (torch.Tensor): Boolean mask of segment tensor

        Returns:
            Loss values
        """
        # since boundaries are rare compared to the number of frames, we upscale the loss on positive boundaries
        # essentially, we penalize missed boundaries to ensure the model doesn't predict a whole lot of nothing
        weight = torch.tensor([self.hparams.pos_weight], device=boundary_logits.device)

        unmasked_boundary_loss = nn.functional.binary_cross_entropy_with_logits(
            boundary_logits, gt_boundaries.float(), pos_weight=weight, reduction='none'
        )
        unmasked_phoneme_frame_loss = nn.functional.cross_entropy(phoneme_logits.permute(0, 2, 1), gt_phonemes, reduction='none')
        unmasked_phoneme_segment_loss = nn.functional.cross_entropy(segment_logits.permute(0, 2, 1), gt_segments, reduction='none')

        b_l = (unmasked_boundary_loss * frame_mask).sum() / frame_mask.sum()
        fp_l = (unmasked_phoneme_frame_loss * frame_mask).sum() / frame_mask.sum()
        sp_l = (unmasked_phoneme_segment_loss * seg_mask).sum() / seg_mask.sum()

        return b_l, fp_l, sp_l

    def _sequential_hit_metric(self, a: torch.Tensor, b: torch.Tensor) -> float:
        pop_mask = torch.zeros(b.size(0), dtype=torch.bool, device=b.device)
        n_hits = 0

        for i in range(a.size(0)):
            for j in range(b.size(0)):
                if not pop_mask[j] and abs(a[i] - b[j]) <= self.hparams.hit_tolerance:
                    n_hits += 1
                    pop_mask[j] = True
                    break

        return n_hits / max(a.size(0), 1)

    @staticmethod
    def _f1_score(precision: torch.Tensor, recall: torch.Tensor) -> torch.Tensor:
        return 2 * precision * recall / (precision + recall + EPSILON)

    @staticmethod
    def _r1_score(precision: float, recall: float) -> float:
        hr = recall * 100
        os = ((recall / precision) - 1) * 100
        r_1 = math.sqrt((100 - hr) ** 2 + os ** 2)
        r_2 = (-os + hr - 100) / math.sqrt(2)

        return 1 - (abs(r_1) + abs(r_2)) / 200

    def _step(self, batch: dict[str, torch.Tensor]) -> StepOutputs:
        mel = batch["mel"]
        bounds = batch["boundaries"]
        frame_lengths = batch["frame_lengths"]
        segment_lengths = batch["segment_lengths"]
        frame_phonemes = batch["frame_phonemes"]
        segment_phonemes = batch["segment_phonemes"]

        # pad mask from original lengths and longest length in batch
        padding_mask = self._make_padding_mask(frame_lengths, mel.size(1))
        valid_mask = ~padding_mask
        segment_padding_mask = self._make_padding_mask(segment_lengths, segment_phonemes.size(1))
        segment_valid_mask = ~segment_padding_mask

        # predict boundaries, phoneme frames, and phoneme segments
        boundary_logits, phoneme_logits, segment_logits = self.forward(mel, padding_mask=padding_mask, gt_boundaries=bounds)

        b_loss, fp_loss, sp_loss = self.compute_loss(boundary_logits, phoneme_logits, segment_logits,
                                 bounds, frame_phonemes, segment_phonemes,
                                 valid_mask, segment_valid_mask)

        boundary_preds = (torch.sigmoid(boundary_logits) >= self.hparams.boundary_threshold).long()
        boundary_acc = ((boundary_preds == bounds) & valid_mask).float().sum() / valid_mask.float().sum()

        precision_sum = 0.0
        recall_sum = 0.0
        for i in range(boundary_preds.size(0)):
            mask = valid_mask[i]
            pred_idx = boundary_preds[i][mask].nonzero(as_tuple=True)[0].float()
            gt_idx = bounds[i][mask].nonzero(as_tuple=True)[0].float()
            precision_sum += self._sequential_hit_metric(pred_idx, gt_idx)
            recall_sum += self._sequential_hit_metric(gt_idx, pred_idx)

        boundary_precision = precision_sum / boundary_preds.size(0)
        boundary_recall = recall_sum / boundary_preds.size(0)
        boundary_r1 = self._r1_score(boundary_precision, boundary_recall)

        frame_phoneme_preds = torch.argmax(phoneme_logits, dim=-1)
        frame_phoneme_acc = ((frame_phoneme_preds == frame_phonemes) & valid_mask).float().sum() / valid_mask.float().sum()

        segment_phoneme_preds = torch.argmax(segment_logits, dim=-1)
        segment_phoneme_acc = ((segment_phoneme_preds == segment_phonemes) & segment_valid_mask).float().sum() / segment_valid_mask.float().sum()

        bound_out = TaskOutput(b_loss, boundary_logits, boundary_preds, boundary_acc, valid_mask, boundary_r1)
        frame_ph_out = TaskOutput(fp_loss, phoneme_logits, frame_phoneme_preds, frame_phoneme_acc, valid_mask, None)
        seg_ph_out = TaskOutput(sp_loss, segment_logits, segment_phoneme_preds, segment_phoneme_acc, segment_valid_mask, None)

        # TODO: these need to be hparams
        b_loss = b_loss * self.hparams.loss_weights[0]
        fp_loss = fp_loss * self.hparams.loss_weights[1]
        sp_loss = sp_loss * self.hparams.loss_weights[2]

        outs = StepOutputs(bound_out, frame_ph_out, seg_ph_out, b_loss + fp_loss + sp_loss)

        return outs

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self._step(batch)

        self.log("train/boundary_loss", outputs.boundary.loss)
        self.log("train/boundary_acc", outputs.boundary.acc)
        self.log("train/boundary_r1", outputs.boundary.r1)

        self.log("train/ph_frame_loss", outputs.frame_phoneme.loss)
        self.log("train/ph_frame_acc", outputs.frame_phoneme.acc)

        self.log("train/ph_seg_loss", outputs.seg_phoneme.loss)
        self.log("train/ph_seg_acc", outputs.seg_phoneme.acc)

        self.log("train/total_loss", outputs.total_loss, prog_bar=True)

        return outputs.total_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self._step(batch)

        self.log("val/boundary_loss", outputs.boundary.loss)
        self.log("val/boundary_acc", outputs.boundary.acc)
        self.log("val/boundary_r1", outputs.boundary.r1)

        self.log("val/ph_frame_loss", outputs.frame_phoneme.loss)
        self.log("val/ph_frame_acc", outputs.frame_phoneme.acc)

        self.log("val/ph_seg_loss", outputs.seg_phoneme.loss)
        self.log("val/ph_seg_acc", outputs.seg_phoneme.acc)

        self.log("val/total_loss", outputs.total_loss, prog_bar=True)

        if batch_idx == 0 and self.logger is not None:
            frame_lengths = batch["frame_lengths"]
            segment_lengths = batch["segment_lengths"]
            for i in range(min(len(batch['mel']), 10)):
                f_lens = frame_lengths[i].item()
                s_lens = segment_lengths[i].item()
                self._log_boundary_visualization(
                    batch['mel'][i][:f_lens], batch['boundaries'][i][:f_lens], outputs.boundary.preds[i][:f_lens], i
                )
                self._log_phoneme_seg_preds(
                    batch['segment_phonemes'][i][:s_lens], outputs.seg_phoneme.preds[i][:s_lens], i
                )

        return outputs.total_loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self._step(batch)

        self.log("test/boundary_loss", outputs.boundary.loss)
        self.log("test/boundary_acc", outputs.boundary.acc)
        self.log("test/boundary_r1", outputs.boundary.r1)

        self.log("test/ph_frame_loss", outputs.frame_phoneme.loss)
        self.log("test/ph_frame_acc", outputs.frame_phoneme.acc)

        self.log("test/ph_seg_loss", outputs.seg_phoneme.loss)
        self.log("test/ph_seg_acc", outputs.seg_phoneme.acc)

        self.log("test/total_loss", outputs.total_loss, prog_bar=True)

    def configure_optimizers(self):
        """Configure muon/adam optimizers"""

        hidden_modules = [self.acoustic, self.temporal.transformer, self.phoneme_classifier]
        hidden_weights = [p for m in hidden_modules for p in m.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for m in hidden_modules for p in m.parameters() if p.ndim < 2]

        nonhidden_params = [
            *self.boundary_classifier.parameters(),
            *(self.temporal.pre_positional_encoding.parameters() if self.temporal.pre_positional_encoding is not None else []),
        ]

        param_groups = [
            dict(params=hidden_weights, use_muon=True, lr=float(self.hparams.muon_lr), weight_decay=float(self.hparams.weight_decay)),
            dict(params=hidden_gains_biases + nonhidden_params, use_muon=False,
                 lr=float(self.hparams.adam_lr), betas=(0.9, 0.95), weight_decay=float(self.hparams.weight_decay)),
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return float(step) / float(max(1, self.hparams.warmup_steps))
            return 1.0

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda
        )

        plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        return (
            [optimizer],
            [
                {"scheduler": warmup_scheduler, "interval": "step"},
                {"scheduler": plateau_scheduler, "monitor": "val/total_loss", "interval": "epoch", "frequency": self.hparams.sch_frequency},
            ],
        )

    def _log_phoneme_seg_preds(self, gt_segs: torch.Tensor, pred_segs: torch.Tensor, i: int) -> None:
        gt_np = gt_segs.detach().cpu().numpy()
        pred_np = pred_segs.detach().cpu().numpy()

        map: dict[int, str] = self.hparams.phoneme_map

        gt_tokens = [map.get(idx) for idx in gt_np if idx]
        pred_tokens = [map.get(idx) for idx in pred_np if idx]

        output_str = f"{' '.join(gt_tokens)} \n {' '.join(pred_tokens)}"

        self.logger.experiment.add_text(f"val/segs_{i}", output_str, self.current_epoch)

    def _log_boundary_visualization(self, mel_spec: torch.Tensor, gt_boundaries: torch.Tensor, pred_boundaries: torch.Tensor, i: int) -> None:
        matplotlib.use("Agg")

        mel_spec_np = mel_spec.detach().cpu().numpy()
        gt_boundaries_np = gt_boundaries.detach().cpu().numpy()
        pred_boundaries_np = pred_boundaries.detach().cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        ax.imshow(
            mel_spec_np.T,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=(0.0, float(mel_spec_np.shape[0]), 0.0, float(mel_spec_np.shape[1])),
        )

        gt_indices = np.where(gt_boundaries_np > 0)[0]
        pred_indices = np.where(pred_boundaries_np > 0)[0]

        for idx in gt_indices:
            ax.axvline(
                x=idx,
                color="lime",
                linewidth=2.0,
                alpha=0.9,
                label="gt" if idx == gt_indices[0] else "",
            )

        for idx in pred_indices:
            ax.axvline(
                x=idx,
                color="red",
                linewidth=1.5,
                alpha=0.9,
                linestyle="--",
                label="pred" if idx == pred_indices[0] else "",
            )

        ax.set_xlabel("frame")
        ax.set_ylabel("mel bin")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

        fig.suptitle(f"Validation Epoch {self.current_epoch}", fontsize=14, y=0.995)
        plt.tight_layout()

        self.logger.experiment.add_figure(f"fig/val_boundaries_{i}", fig, self.current_epoch)
        plt.close(fig)

    def export(self, path: str) -> None:
        """
        Export current model state as ONNX file

        Args:
            path (str): Path to save ONNX model
        """
        self.cpu()

        torch.onnx.export(
            MinaONNXWrapper(self),
            (self.example_input_array,),
            path,
            input_names=["mel"],
            output_names=["boundaries", "phonemes"],
            dynamic_shapes={"x": {1: "y"}},
            opset_version=None,
            external_data=False,
        )

class MinaONNXWrapper(nn.Module):
    """ONNX-friendly wrapper for Mina. Applies threshold to forward pass"""
    def __init__(self, model: MINA):
        super().__init__()
        self.model = model
        self.boundary_threshold = model.hparams.boundary_threshold

    def forward(self, x: torch.Tensor) -> tuple[Tensor, Tensor]:
        boundary_logits, phoneme_logits, _ = self.model(x)
        boundaries = (torch.sigmoid(boundary_logits) >= self.boundary_threshold).long()
        phonemes = torch.argmax(phoneme_logits, dim=-1)
        return boundaries, phonemes

class StepOutputs(NamedTuple):
    """All outputs for a single training step inside of MINA"""
    boundary: TaskOutput
    frame_phoneme: TaskOutput
    seg_phoneme: TaskOutput
    total_loss: torch.Tensor

class TaskOutput(NamedTuple):
    """Outputs for a single MINA task"""
    loss: torch.Tensor
    logits: torch.Tensor
    preds: torch.Tensor
    acc: torch.Tensor
    valid_mask: torch.Tensor
    r1: float | None

