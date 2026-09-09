import os
from pathlib import Path
from typing import Any

import lightning
import onnx
import onnxslim
import torch
import argparse

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichModelSummary, BatchSizeFinder
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.core.saving import save_hparams_to_yaml
from omegaconf import OmegaConf

from mina.dataset import MinaDataModule
from mina.model import MINA
from mina.positional_encoding import PositionalEncodingType

if __name__ == '__main__':
    torch.serialization.add_safe_globals([PositionalEncodingType])
    torch._dynamo.config.capture_scalar_outputs = True
    torch.set_float32_matmul_precision('medium')

    config = OmegaConf.load("config/config.yaml")
    cli = OmegaConf.from_cli()
    cfg = OmegaConf.merge(config, cli)

    bin_data = Path(cfg.core.bin_data_dir)
    lightning.seed_everything(72)

    data_module = MinaDataModule(bin_data, cfg.train.batch_size if cfg.train.batch_size is not None else 1, cfg.train.workers)
    model = MINA(
        mel_dim=cfg.model.mel_dim,
        latent_dim=cfg.model.latent_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_conv_layers=cfg.model.num_conv_layers,
        num_phoneme_layers=cfg.model.num_phoneme_layers,
        num_boundary_layers=cfg.model.num_boundary_layers,
        num_conv_heads=cfg.model.num_conv_heads,
        num_phoneme_heads=cfg.model.num_phoneme_heads,
        num_boundary_heads=cfg.model.num_boundary_heads,
        phoneme_feedforward_dim=cfg.model.phoneme_feedforward_dim,
        boundary_feedforward_dim=cfg.model.boundary_feedforward_dim,
        conv_dropout=cfg.model.conv_dropout,
        phoneme_dropout=cfg.model.phoneme_dropout,
        boundary_dropout=cfg.model.boundary_dropout,
        phoneme_classifier_dropout=cfg.model.phoneme_classifier_dropout,
        kernel_size=cfg.model.kernel_size,
        max_len=data_module.rec_max_len,
        sr=cfg.core.sample_rate,
        hop_length=cfg.core.n_fft // 4,
        muon_lr=cfg.train.muon_lr,
        adam_lr=cfg.train.adam_lr,
        pos_weight=cfg.train.pos_weight,
        boundary_threshold=cfg.train.boundary_threshold,
        pe_type=cfg.train.pe_type,
        vocab_size=data_module.vocab_size,
        weight_decay=cfg.train.weight_decay,
        warmup_steps=cfg.train.warmup_steps,
        sch_frequency=cfg.train.val_n_epochs,
        hit_tolerance=cfg.train.hit_tolerance,
        phoneme_map=data_module.phoneme_map,
        loss_weights=(cfg.train.boundary_loss_weight, cfg.train.phoneme_frame_loss_weight, cfg.train.phoneme_segment_loss_weight),
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="{epoch:02d}-{step:02d}",
        save_top_k=1,
        monitor="val/total_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val/total_loss",
        patience=10,
        mode='min',
        verbose=True
    )

    logger = TensorBoardLogger(".", name="lightning_logs")

    callbacks: list[Any] = [checkpoint_callback, early_stop_callback, RichModelSummary(max_depth=1)]

    if cfg.train.batch_size is None:
        callbacks.append(BatchSizeFinder(mode="binsearch", margin=0.6))

    trainer = lightning.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        log_every_n_steps=10,
        check_val_every_n_epoch=5,
        precision="16-mixed",

    )

    trainer.fit(
        model,
        datamodule=data_module,
    )

    trainer.test(model, datamodule=data_module, ckpt_path="best")

    best_model_path = checkpoint_callback.best_model_path
    best_model_dir = os.path.dirname(best_model_path)
    onnx_path = os.path.join(best_model_dir, "mina.onnx")
    hparams_path = os.path.join(best_model_dir, "hparams.yaml")

    best_model = MINA.load_from_checkpoint(checkpoint_path=best_model_path)
    best_model.eval()
    best_model.export(onnx_path)
    onnx.save(onnxslim.slim(onnx.load(onnx_path)), onnx_path)
    save_hparams_to_yaml(config_yaml=hparams_path, hparams=trainer.model.hparams)
