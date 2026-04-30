from typing import NamedTuple

import lightning
import numpy as np
import torch
from muon import SingleDeviceMuonWithAuxAdam
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mina.acoustic import ConvolutionalAcousticEncoder
from mina.boundary import BoundaryDetector

import matplotlib
import matplotlib.pyplot as plt

from mina.positional_encoding import PositionalEncodingType

EPSILON = 1e-8

class MINA(lightning.LightningModule):
    def __init__(self, d_mel: int, d_l: int, d_h: int, conv_layers: int,
                 num_heads: int, tf_layers: int, tf_dim_ff: int, dropout_conv: float,
                 dropout_tf: float, kernel_size: int, max_len: int, sr: int,
                 hop_length: int, muon_lr: float, adam_lr: float, pos_weight: float,
                 boundary_threshold: float, pe_type: PositionalEncodingType,
                 weight_decay: float, warmup_steps: int):
        super().__init__()
        self.save_hyperparameters()

        self.acoustic = ConvolutionalAcousticEncoder(d_mel, d_l, d_h, conv_layers, kernel_size, dropout_conv)
        self.detector = BoundaryDetector(d_h, num_heads, tf_layers, tf_dim_ff, dropout_tf, max_len, pe_type)
        # TODO self.classifier = PhonemeClassifier(whatever)

        # for plottage
        self.sr = sr
        self.hop_length = hop_length

    def forward(self, x: torch.Tensor, padding_mask=None) -> torch.Tensor:
        x = self.acoustic(x)
        x = self.detector(x, padding_mask=padding_mask)
        return x

    def on_train_start(self):
        self.acoustic.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
        self.detector.compile(mode="max-autotune-no-cudagraphs", dynamic=True)

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

    def compute_loss(self, logits: torch.Tensor, boundaries: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Binary cross entropy loss with mask and weighting

        Args:
            logits (torch.Tensor): Tensor of logits
            boundaries (torch.Tensor): Tensor of boundaries
            mask (torch.Tensor): Boolean mask tensor

        Returns:
            Loss values
        """
        # since boundaries are rare compared to the number of frames, we upscale the loss on positive boundaries
        # essentially, we penalize missed boundaries to ensure the model doesn't predict a whole lot of nothing
        weight = torch.tensor([self.hparams.pos_weight], device=logits.device)

        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, boundaries.float(), pos_weight=weight, reduction='none'
        )

        # exclude the padding positions from the final loss computation
        return (loss * mask).sum() / mask.sum()

    @staticmethod
    def _precision_recall_f1(counts: DetectionCounts) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates precision, recall and f1 score

        Args:
            counts: True positives, false positives, and false negatives

        Returns:
            Tuple of precision, recall, f1
        """

        precision = counts.true_positives / (counts.true_positives + counts.false_positives + EPSILON)
        recall = counts.true_positives / (counts.true_positives + counts.false_negatives + EPSILON)
        f1 = 2 * precision * recall / (precision + recall + EPSILON)
        return precision, recall, f1

    def _step(self, batch: dict[str, torch.Tensor]) -> StepOutputs:
        mel, bounds, lengths, phonemes = batch['mel'], batch['boundaries'], batch['lengths'], batch['phonemes']

        # pad mask from original lengths and longest length in batch
        padding_mask = self._make_padding_mask(lengths, mel.size(1))
        valid_mask = ~padding_mask

        logits = self.forward(mel, padding_mask=padding_mask)
        loss = self.compute_loss(logits, bounds, valid_mask.float())

        probs = torch.sigmoid(logits)
        preds = (probs >= self.hparams.boundary_threshold).long()
        acc = ((preds == bounds) & valid_mask).float().sum() / valid_mask.float().sum()

        # tp - frames predicted as a boundary that are actually boundaries
        # fp - frames predicted as a boundary that are not boundaries
        # fn - frames predicted as a non-boundary that are boundaries
        counts = DetectionCounts(
            true_positives=((preds == 1) & (bounds == 1) & valid_mask).float().sum(),
            false_positives=((preds == 1) & (bounds == 0) & valid_mask).float().sum(),
            false_negatives=((preds == 0) & (bounds == 1) & valid_mask).float().sum(),
        )

        return StepOutputs(loss=loss, logits=logits, probs=probs, acc=acc, valid_mask=valid_mask, counts=counts)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self._step(batch)
        precision, recall, f1 = self._precision_recall_f1(outputs.counts)

        self.log("train/loss", outputs.loss, on_step=True, on_epoch=True)
        self.log("train/acc", outputs.acc, on_step=True, on_epoch=True)
        self.log("train/precision", precision, on_step=False, on_epoch=True)
        self.log("train/recall", recall, on_step=False, on_epoch=True)
        self.log("train/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return outputs.loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self._step(batch)
        precision, recall, f1 = self._precision_recall_f1(outputs.counts)

        self.log("val/loss", outputs.loss, prog_bar=True)
        self.log("val/acc", outputs.acc)
        self.log("val/precision", precision, prog_bar=True)
        self.log("val/recall", recall)
        self.log("val/f1", f1, prog_bar=True)

        if batch_idx == 0 and self.logger is not None:
            lengths = batch['lengths']
            for i in range(len(batch['mel'])):
                lens = lengths[i].item()
                self._log_boundary_visualization(
                    batch['mel'][i][:lens], batch['boundaries'][i][:lens], outputs.probs[i][:lens], i
                )

        return outputs.loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self._step(batch)
        precision, recall, f1 = self._precision_recall_f1(outputs.counts)

        self.log("test/loss", outputs.loss)
        self.log("test/acc", outputs.acc)
        self.log("test/precision", precision)
        self.log("test/recall", recall)
        self.log("test/f1", f1)

    def configure_optimizers(self):
        """Configure muon/adam optimizers"""

        hidden_modules = [self.acoustic, self.detector.transformer]
        hidden_weights = [p for m in hidden_modules for p in m.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for m in hidden_modules for p in m.parameters() if p.ndim < 2]

        nonhidden_params = [
            *self.detector.output.parameters(),
            *self.detector.positional_encoding.parameters(),
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
                {"scheduler": plateau_scheduler, "monitor": "val/f1", "interval": "epoch"},
            ],
        )

    def _log_boundary_visualization(self, mel_spec: torch.Tensor, gt_boundaries: torch.Tensor, pred_probs: torch.Tensor, i: int) -> None:
        matplotlib.use("Agg")

        mel_spec_np = mel_spec.detach().cpu().numpy()
        gt_boundaries_np = gt_boundaries.detach().cpu().numpy()
        pred_probs_np = pred_probs.detach().cpu().numpy()
        pred_boundaries_np = (pred_probs_np >= self.hparams.boundary_threshold).astype(int)

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
            output_names=["boundaries"],
            dynamic_shapes={"x": {1: "seq_len"}},
            opset_version=None,
            external_data=False,
        )

class MinaONNXWrapper(nn.Module):
    """ONNX-friendly wrapper for Mina. Applies threshold to forward pass"""
    def __init__(self, model: MINA):
        super().__init__()
        self.model = model
        self.threshold = model.hparams.boundary_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return torch.sigmoid(logits) >= self.threshold

class DetectionCounts(NamedTuple):
    """Detection counts for calculating F1 score"""
    true_positives: torch.Tensor
    false_positives: torch.Tensor
    false_negatives: torch.Tensor

class StepOutputs(NamedTuple):
    """All outputs for a single training step inside of MINA"""
    loss: torch.Tensor
    logits: torch.Tensor
    probs: torch.Tensor
    acc: torch.Tensor
    valid_mask: torch.Tensor
    counts: DetectionCounts
