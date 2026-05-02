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
                 phoneme_map: dict[int, str]):
        super().__init__()
        self.save_hyperparameters()

        self.acoustic = ConvAcousticEncoder(d_mel, d_l, d_h, conv_layers, kernel_size, dropout_conv)
        self.temporal = TemporalContextEncoder(d_h, num_heads, tf_layers, tf_dim_ff, dropout_tf, max_len, pe_type)
        self.boundary_classifier = nn.Linear(d_h, 1)
        self.phoneme_classifier = PhonemeClassifier(d_h, vocab_size, phoneme_dropout)


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

    @staticmethod
    def _f1_score(counts: DetectionCounts) -> torch.Tensor:
        """
        Calculates precision, recall and f1 score

        Args:
            counts: True positives, false positives, and false negatives

        Returns:
            F1 score
        """
        precision = counts.true_positives / (counts.true_positives + counts.false_positives + EPSILON)
        recall = counts.true_positives / (counts.true_positives + counts.false_negatives + EPSILON)
        f1 = 2 * precision * recall / (precision + recall + EPSILON)
        return f1

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

        frame_phoneme_preds = torch.argmax(phoneme_logits, dim=-1)
        frame_phoneme_acc = ((frame_phoneme_preds == frame_phonemes) & valid_mask).float().sum() / valid_mask.float().sum()

        segment_phoneme_preds = torch.argmax(segment_logits, dim=-1)
        segment_phoneme_acc = ((segment_phoneme_preds == segment_phonemes) & segment_valid_mask).float().sum() / segment_valid_mask.float().sum()

        # tp - frames predicted as a boundary that are actually boundaries
        # fp - frames predicted as a boundary that are not boundaries
        # fn - frames predicted as a non-boundary that are boundaries
        bound_counts = DetectionCounts(
            true_positives=((boundary_preds == 1) & (bounds == 1) & valid_mask).float().sum(),
            false_positives=((boundary_preds == 1) & (bounds == 0) & valid_mask).float().sum(),
            false_negatives=((boundary_preds == 0) & (bounds == 1) & valid_mask).float().sum(),
        )

        bound_out = TaskOutput(b_loss, boundary_logits, boundary_preds, boundary_acc, valid_mask, bound_counts)
        frame_ph_out = TaskOutput(fp_loss, phoneme_logits, frame_phoneme_preds, frame_phoneme_acc, valid_mask, None)
        seg_ph_out = TaskOutput(sp_loss, segment_logits, segment_phoneme_preds, segment_phoneme_acc, segment_valid_mask, None)

        # TODO: these need to be hparams
        b_loss = b_loss * 1.0
        fp_loss = fp_loss * 0.3
        sp_loss = sp_loss * 0.3

        outs = StepOutputs(bound_out, frame_ph_out, seg_ph_out, b_loss + fp_loss + sp_loss)

        return outs

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self._step(batch)
        f1 = self._f1_score(outputs.boundary.counts)

        self.log("train/boundary_loss", outputs.boundary.loss, on_step=True, on_epoch=True)
        self.log("train/boundary_acc", outputs.boundary.acc, on_step=True, on_epoch=True)
        self.log("train/boundary_f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        self.log("train/ph_frame_loss", outputs.frame_phoneme.loss, on_step=True, on_epoch=True)
        self.log("train/ph_frame_acc", outputs.frame_phoneme.acc, on_step=True, on_epoch=True)

        self.log("train/seg_phoneme_loss", outputs.seg_phoneme.loss, on_step=True, on_epoch=True)
        self.log("train/seg_ph_acc", outputs.seg_phoneme.acc, on_step=True, on_epoch=True)

        self.log("train/total_loss", outputs.total_loss, on_step=True, on_epoch=True)

        return outputs.total_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self._step(batch)
        f1 = self._f1_score(outputs.boundary.counts)

        self.log("val/boundary_loss", outputs.boundary.loss, on_step=True, on_epoch=True)
        self.log("val/boundary_acc", outputs.boundary.acc, on_step=True, on_epoch=True)
        self.log("val/boundary_f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        self.log("val/ph_frame_loss", outputs.frame_phoneme.loss, on_step=True, on_epoch=True)
        self.log("val/ph_frame_acc", outputs.frame_phoneme.acc, on_step=True, on_epoch=True)

        self.log("val/seg_phoneme_loss", outputs.seg_phoneme.loss, on_step=True, on_epoch=True)
        self.log("val/seg_ph_acc", outputs.seg_phoneme.acc, on_step=True, on_epoch=True)

        self.log("val/total_loss", outputs.total_loss, on_step=True, on_epoch=True)

        if batch_idx == 0 and self.logger is not None:
            frame_lengths = batch["frame_lengths"]
            for i in range(min(len(batch['mel']), 10)):
                f_lens = frame_lengths[i].item()
                self._log_boundary_visualization(
                    batch['mel'][i][:f_lens], batch['boundaries'][i][:f_lens], outputs.boundary.preds[i][:f_lens], i
                )
                self._log_segment_visualization(
                    batch['segment_phonemes'][i], outputs.seg_phoneme.preds[i], i
                )

        return outputs.total_loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self._step(batch)
        f1 = self._f1_score(outputs.boundary.counts)

        self.log("test/boundary_loss", outputs.boundary.loss, on_step=True, on_epoch=True)
        self.log("test/boundary_acc", outputs.boundary.acc, on_step=True, on_epoch=True)
        self.log("test/boundary_f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        self.log("test/ph_frame_loss", outputs.frame_phoneme.loss, on_step=True, on_epoch=True)
        self.log("test/ph_frame_acc", outputs.frame_phoneme.acc, on_step=True, on_epoch=True)

        self.log("test/seg_phoneme_loss", outputs.seg_phoneme.loss, on_step=True, on_epoch=True)
        self.log("test/seg_ph_acc", outputs.seg_phoneme.acc, on_step=True, on_epoch=True)

        self.log("test/total_loss", outputs.total_loss, on_step=True, on_epoch=True)

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

        gt_tokens = [map.get(idx) for idx in gt_np if idx != 0]
        pred_tokens = [map.get(idx) for idx in pred_np if idx != 0]

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

        self.logger.experiment.add_figure(f"val/boundaries_{i}", fig, self.current_epoch)
        plt.close(fig)

    def export(self, path: str) -> None:
        """
        Export current model state as ONNX file

        Args:
            path (str): Path to save ONNX model
        """
        self.cpu()

        dummy_mel = torch.zeros(1, self.hparams.max_len, self.acoustic.mel_dim)

        torch.onnx.export(
            MinaONNXWrapper(self),
            (dummy_mel,),
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

class DetectionCounts(NamedTuple):
    """Detection counts for calculating F1 score"""
    true_positives: torch.Tensor
    false_positives: torch.Tensor
    false_negatives: torch.Tensor

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
    counts: DetectionCounts | None

