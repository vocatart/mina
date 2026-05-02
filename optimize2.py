import argparse
import gc
import os
from pathlib import Path
import tempfile

os.environ.setdefault("RAY_AIR_NEW_OUTPUT", "0")

import lightning
import torch
import ray
import ray.train
from ray import tune
from ray.tune import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from lightning.pytorch.callbacks import EarlyStopping
import optuna

from binarize import Preprocessor
from mina.dataset import MinaDataModule
from mina.model import MINA
from mina.positional_encoding import PositionalEncodingType

METRICS = [
    "val/total_loss",
    "val/boundary_loss",
    "val/ph_frame_loss",
    "val/ph_seg_loss",
    "val/boundary_acc",
    "val/ph_frame_acc",
    "val/ph_seg_acc",
    "val/boundary_f1",
]
MODES = ["min", "min", "min", "min", "max", "max", "max", "max"]
PRIMARY_METRIC = "val/total_loss"
PRIMARY_MODE = "min"


def trainable(config: dict):
    try:
        sr = config["sr"]
        mels = config["mels"]
        hop_length = config["hop_length"]
        n_fft = config["n_fft"]
        conv_dim = config["conv_dim"]
        num_conv = config["num_conv"]
        num_heads = config["num_heads"]
        dim_multiplier = config["dim_multiplier"]
        kernel_size = config["kernel_size"]
        conv_dropout = config["conv_dropout"]
        tf_layers = config["tf_layers"]
        ff_multiplier = config["ff_multiplier"]
        transformer_dropout = config["transformer_dropout"]
        phoneme_dropout = config["phoneme_dropout"]
        thresh = config["thresh"]
        muon_lr = config["muon_lr"]
        adam_lr = config["adam_lr"]
        pos_weight = config["pos_weight"]
        weight_decay = config["weight_decay"]
        warmup_steps = config["warmup_steps"]

        latent_dim = num_heads * dim_multiplier * 8
        tf_dim_ff = latent_dim * ff_multiplier

        temp_dir = tempfile.TemporaryDirectory()
        temp_bin_dir = Path(temp_dir.name) / "bin"

        proc = Preprocessor(argparse.Namespace(**{
            "dataset": config["data_dir"],
            "output": temp_bin_dir,
            "sr": sr,
            "mels": mels,
            "hop_length": hop_length,
            "n_fft": n_fft,
            "val_split": 0.10,
            "time_split": 10,
            "audio_types": ["flac"],
            "workers": 20,
        }))

        proc.process_audio()
        proc.save_metadata()

        data_module = MinaDataModule(temp_bin_dir, config["batch_size"], config["workers"])
        model = MINA(
            d_mel=mels,
            d_l=conv_dim,
            d_h=latent_dim,
            conv_layers=num_conv,
            num_heads=num_heads,
            tf_layers=tf_layers,
            tf_dim_ff=tf_dim_ff,
            kernel_size=kernel_size,
            dropout_conv=conv_dropout,
            dropout_tf=transformer_dropout,
            muon_lr=muon_lr,
            adam_lr=adam_lr,
            weight_decay=weight_decay,
            pos_weight=pos_weight,
            max_len=data_module.rec_max_len,
            sr=sr,
            hop_length=hop_length,
            boundary_threshold=thresh,
            pe_type=PositionalEncodingType.from_str("rope"),
            warmup_steps=warmup_steps,
            do_compile=False,
            phoneme_dropout=phoneme_dropout,
            phoneme_map=data_module.phoneme_map,
            vocab_size=data_module.vocab_size,
            sch_frequency=1,
            loss_weights=(1.0, 1.0, 1.0),
        )

        tune_callback = TuneReportCallback(
            {m: m for m in METRICS},
            on="validation_end",
        )

        early_stop_callback = EarlyStopping(
            monitor="val/total_loss",
            patience=10,
            mode="min",
        )

        trainer = lightning.Trainer(
            accelerator="auto",
            devices="auto",
            callbacks=[tune_callback, early_stop_callback],
            logger=True,
            gradient_clip_val=1.0,
            accumulate_grad_batches=1,
            log_every_n_steps=10,
            check_val_every_n_epoch=1,
            precision="16-mixed",
            max_epochs=150,
            enable_progress_bar=False,
        )

        trainer.fit(model, data_module)
        temp_dir.cleanup()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("pruning oom trial")
            torch.cuda.empty_cache()
            gc.collect()
            try:
                del model
            except NameError:
                pass
            ray.train.report({
                m: float("inf") if mode == "min" else 0.0
                for m, mode in zip(METRICS, MODES)
            })
        else:
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trials", type=int, default=500)
    args = parser.parse_args()

    param_space = {
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "workers": args.workers,
        # dataset
        "sr": tune.choice([8000, 11025, 22050, 32000, 44100]),
        "mels": tune.choice([40, 64, 80, 128]),
        "hop_length": tune.choice([160, 256]),
        "n_fft": tune.choice([400, 512, 1024]),
        # conv
        "conv_dim": tune.choice([64, 128, 256]),
        "num_conv": tune.randint(2, 7),
        "num_heads": tune.choice([2, 4, 8]),
        "dim_multiplier": tune.randint(2, 7),
        "kernel_size": tune.choice([3, 5, 7]),
        "conv_dropout": tune.quniform(0.0, 0.5, 0.05),
        # tf
        "tf_layers": tune.randint(1, 5),
        "ff_multiplier": tune.choice([1, 2, 4]),
        "transformer_dropout": tune.quniform(0.1, 0.5, 0.05),
        # ph
        "phoneme_dropout": tune.quniform(0.1, 0.5, 0.05),
        # training
        "thresh": tune.quniform(0.3, 0.7, 0.05),
        "muon_lr": tune.loguniform(1e-4, 1e-2),
        "adam_lr": tune.loguniform(1e-4, 1e-2),
        "pos_weight": tune.uniform(-2.0, 2.0),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
        "warmup_steps": tune.choice([0, 100, 200, 300, 400, 500]),
    }

    search_alg = OptunaSearch(
        metric=METRICS,
        mode=MODES,
        sampler=optuna.samplers.NSGAIISampler(),
        storage=optuna.storages.RDBStorage("sqlite:///db.sqlite3"),
        study_name="mina",
    )

    scheduler = ASHAScheduler(
        metric=PRIMARY_METRIC,
        mode=PRIMARY_MODE,
        max_t=150,
        grace_period=10,
        reduction_factor=3,
    )

    ray.init(log_to_driver=False)

    tuner = tune.Tuner(
        trainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=args.trials,
            max_concurrent_trials=1,
        ),
        run_config=RunConfig(
            name="mina",
            storage_path=str(Path("ray_results").resolve()),
        ),
    )

    results = tuner.fit()

    df = results.get_dataframe(filter_metric=PRIMARY_METRIC, filter_mode=PRIMARY_MODE)
    print(f"\nCompleted trials: {len(df)}")

    print("\nTop 5 by val/total_loss:")
    cols = ["config/muon_lr", "config/adam_lr", "config/sr", "config/mels",
            "val/total_loss", "val/boundary_f1", "val/ph_seg_acc"]
    print(df.nsmallest(5, PRIMARY_METRIC)[cols].to_string())

    print("\nTop 5 by val/boundary_f1:")
    print(df.nlargest(5, "val/boundary_f1")[cols].to_string())
