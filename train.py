import os
from pathlib import Path

import lightning
import onnx
import onnxslim
import torch
import argparse

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichModelSummary
from lightning.pytorch.loggers import TensorBoardLogger

from mina.dataset import MinaDataModule
from mina.model import MINA
from mina.positional_encoding import PositionalEncodingType

if __name__ == '__main__':
    torch.serialization.add_safe_globals([PositionalEncodingType])
    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir", type=str)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--conv_dim", type=int, default=256) # d_l
    parser.add_argument("--latent_dim", type=int, default=64) # d_h
    parser.add_argument("--num_conv", type=int, default=5)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--tf_layers", type=int, default=4)
    parser.add_argument("--tf_dim_ff", type=int, default=256)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--conv_dropout", type=float, default=0.25)
    parser.add_argument("--transformer_dropout", type=float, default=0.05)
    parser.add_argument("--thresh", type=float, default=0.6)
    parser.add_argument(
        "--pe_type",
        type=PositionalEncodingType,
        choices=["sinusoidal", "learned", "rope"],
        default=PositionalEncodingType.ROPE,
    )

    parser.add_argument("--lr_muon", type=float, default=1.7e-3)
    parser.add_argument("--lr_adam", type=float, default=8.1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--pos_weight", type=float, default=1.9)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--val_n_epochs", type=int, default=10)
    parser.add_argument("--no_compile", action="store_false")

    args = parser.parse_args()
    bin_data = Path(args.data_dir)
    lightning.seed_everything(76_805)

    data_module = MinaDataModule(bin_data, args.batch_size, args.num_workers)
    model = MINA(
        d_mel=data_module.n_mels,
        d_l=args.conv_dim,
        d_h=args.latent_dim,
        conv_layers=args.num_conv,
        num_heads=args.num_heads,
        tf_layers=args.tf_layers,
        tf_dim_ff=args.tf_dim_ff,
        kernel_size=args.kernel_size,
        dropout_conv=args.conv_dropout,
        dropout_tf=args.transformer_dropout,
        muon_lr=args.lr_muon,
        adam_lr=args.lr_adam,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        max_len=data_module.rec_max_len,
        sr=data_module.sr,
        hop_length=data_module.hop_length,
        boundary_threshold=args.thresh,
        pe_type=args.pe_type,
        warmup_steps=args.warmup_steps,
        sch_frequency=args.val_n_epochs,
        compile=not args.no_compile
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="{epoch:02d}-{step:02d}",
        save_top_k=1,
        monitor="val/f1",
        mode="max",
    )

    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=3,
        mode='min',
        verbose=True
    )

    logger = TensorBoardLogger(".", name="lightning_logs")

    trainer = lightning.Trainer(
        max_epochs=args.num_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback, early_stop_callback, RichModelSummary(max_depth=2)],
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

    best_model = MINA.load_from_checkpoint(checkpoint_path=best_model_path)
    best_model.eval()
    best_model.export(onnx_path)
    onnx.save(onnxslim.slim(onnx.load(onnx_path)), onnx_path)
