import lightning
import numpy as np
import torch
from muon import SingleDeviceMuonWithAuxAdam
from torch import nn
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ReduceLROnPlateau

from mina.acoustic import ConvolutionalAcousticEncoder
from mina.boundary import BoundaryDetector

import matplotlib
import matplotlib.pyplot as plt

from mina.positional_encoding import PositionalEncodingType


class MINA(lightning.LightningModule):
    def __init__(self, d_mel, d_l, d_h, conv_layers,
        num_heads, tf_layers, tf_dim_ff, dropout_conv, dropout_tf, kernel_size,
        max_len, sr, hop_length, muon_lr, adam_lr, pos_weight, boundary_threshold,
        pe_type: PositionalEncodingType, weight_decay, warmup_steps):
        super().__init__()
        self.save_hyperparameters(ignore=["sr", "hop_length"])

        self.acoustic = ConvolutionalAcousticEncoder(d_mel, d_l, d_h, conv_layers, kernel_size, dropout_conv)
        self.detector = BoundaryDetector(d_h, num_heads, tf_layers, tf_dim_ff, dropout_tf, max_len, pe_type)
        # TODO self.classifier = PhonemeClassifier(whatever)

        # for plottage
        self.sr = sr
        self.hop_length = hop_length

    def forward(self, x, padding_mask=None):
        x = self.acoustic(x)
        return self.detector(x, padding_mask=padding_mask)

    @staticmethod
    def _make_padding_mask(lengths, max_len):
        """Creates a boolean padding mask for positions that are padded"""
        idx = torch.arange(max_len, device=lengths.device)
        return idx.unsqueeze(0) >= lengths.unsqueeze(1)

    def compute_loss(self, logits, boundaries, mask):
        # since boundaries are rare compared to the number of frames, we upscale the loss on positive boundaries
        # essentially, we penalize missed boundaries to ensure the model doesn't predict a whole lot of nothing
        weight = torch.tensor([self.hparams.pos_weight], device=logits.device)

        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, boundaries.float(), pos_weight=weight, reduction='none'
        )

        # exclude the padding positions from the final loss computation
        return (loss * mask).sum() / mask.sum()

    @staticmethod
    def _precision_recall_f1(tp, fp, fn):
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f1

    def _step(self, batch):
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
        tp = ((preds == 1) & (bounds == 1) & valid_mask).float().sum()
        fp = ((preds == 1) & (bounds == 0) & valid_mask).float().sum()
        fn = ((preds == 0) & (bounds == 1) & valid_mask).float().sum()

        return logits, loss, acc, valid_mask, probs, tp, fp, fn

    def training_step(self, batch, batch_idx):
        _, loss, acc, _, _, tp, fp, fn = self._step(batch)
        precision, recall, f1 = self._precision_recall_f1(tp, fp, fn)

        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True)
        self.log("train/precision", precision, on_step=False, on_epoch=True)
        self.log("train/recall", recall, on_step=False, on_epoch=True)
        self.log("train/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits, loss, acc, _, probs, tp, fp, fn = self._step(batch)
        precision, recall, f1 = self._precision_recall_f1(tp, fp, fn)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc)
        self.log("val/precision", precision, prog_bar=True)
        self.log("val/recall", recall)
        self.log("val/f1", f1, prog_bar=True)

        if batch_idx == 0 and self.logger is not None:
            lengths = batch['lengths']
            for i in range(len(batch['mel'])):
                L = lengths[i].item()
                self._log_boundary_visualization(
                    batch['mel'][i][:L], batch['boundaries'][i][:L], probs[i][:L], i
                )

        return loss

    def test_step(self, batch, batch_idx):
        _, loss, acc, _, _, tp, fp, fn = self._step(batch)
        precision, recall, f1 = self._precision_recall_f1(tp, fp, fn)

        self.log("test/loss", loss)
        self.log("test/acc", acc)
        self.log("test/precision", precision)
        self.log("test/recall", recall)
        self.log("test/f1", f1)

    def configure_optimizers(self):
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

        return {
            "optimizer": optimizer,
            "lr_schedulers": [
                {
                    "scheduler": warmup_scheduler,
                    "interval": "step",
                },
                {
                    "scheduler": plateau_scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                },
            ],
        }

    def _log_boundary_visualization(self, mel_spec, gt_boundaries, pred_probs, i):
        matplotlib.use('Agg')

        mel_spec_np = mel_spec.detach().cpu().numpy()
        gt_boundaries_np = gt_boundaries.detach().cpu().numpy()
        pred_probs_np = pred_probs.detach().cpu().numpy()
        pred_boundaries_np = (pred_probs_np >= self.hparams.boundary_threshold).astype(int)

        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        ax.imshow(
            mel_spec_np.T,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            extent=[0, mel_spec_np.shape[0], 0, mel_spec_np.shape[1]],
        )

        gt_indices = np.where(gt_boundaries_np > 0)[0]
        pred_indices = np.where(pred_boundaries_np > 0)[0]

        for idx in gt_indices:
            ax.axvline(
                x=idx,
                color='lime',
                linewidth=2.0,
                alpha=0.9,
                label='gt' if idx == gt_indices[0] else '',
            )

        for idx in pred_indices:
            ax.axvline(
                x=idx,
                color='red',
                linewidth=1.5,
                alpha=0.9,
                linestyle='--',
                label='pred' if idx == pred_indices[0] else '',
            )

        ax.set_xlabel('frame')
        ax.set_ylabel('mel bin')
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)

        fig.suptitle(f'Validation Epoch {self.current_epoch}', fontsize=14, y=0.995)
        plt.tight_layout()

        self.logger.experiment.add_figure(f'val/boundaries_{i}', fig, self.current_epoch)
        plt.close(fig)
