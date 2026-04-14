from pathlib import Path

import lightning
import torch
import argparse

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from mina.dataset import MinaDataModule
from mina.model import MINA
from mina.positional_encoding import PositionalEncoding, PositionalEncodingType

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir", type=str)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--conv_dim", type=int, default=256) # d_l
    parser.add_argument("--latent_dim", type=int, default=192) # d_h
    parser.add_argument("--num_conv", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--tf_layers", type=int, default=4)
    parser.add_argument("--tf_dim_ff", type=int, default=768)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--conv_dropout", type=float, default=0.1)
    parser.add_argument("--transformer_dropout", type=float, default=0.1)
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument(
        "--pe_type",
        type=PositionalEncodingType,
        choices=list(PositionalEncodingType),
        default=PositionalEncodingType.SINUSOIDAL,
        help="Type of positional encoding"
    )

    parser.add_argument("--lr_muon", type=float, default=5e-4)
    parser.add_argument("--lr_adam", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--pos_weight", type=float, default=5.0)
    parser.add_argument("--warmup_steps", type=int, default=500)

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
        warmup_steps=args.warmup_steps
    )
    model.compile(mode="max-autotune-no-cudagraphs", dynamic=True)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="{epoch:02d}-{step:02d}",
        save_top_k=5,
        monitor="val/f1",
        mode="max",
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor="val/f1",
        patience=10,
        mode='max',
        verbose=True
    )

    logger = TensorBoardLogger("./logs", name="segment")

    trainer = lightning.Trainer(
        max_epochs=args.num_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        precision='32'
    )

    trainer.fit(
        model,
        datamodule=data_module,
    )

    trainer.test(model, datamodule=data_module, ckpt_path="best")
