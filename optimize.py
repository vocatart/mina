import argparse
import gc
from pathlib import Path

import lightning
import optuna
import torch
from optuna.integration import PyTorchLightningPruningCallback
from lightning.pytorch.callbacks import EarlyStopping

from binarize import Preprocessor
import tempfile

from mina.dataset import MinaDataModule
from mina.model import MINA
from mina.positional_encoding import PositionalEncodingType


def objective(trial: optuna.trial.Trial, data_dir: Path, batch_size: int, workers: int) -> float:
    try:
        # dataset hyperparameters
        # fft has a constraint that it must be higher than the hop length
        sr = trial.suggest_categorical("sr", [8000, 11025, 22050, 32000, 44100])
        mels = trial.suggest_categorical("mels", [40, 64, 80, 128])
        hop_length = trial.suggest_categorical("hop_length", [160, 256])
        n_fft = trial.suggest_categorical("n_fft", [f for f in [400, 512, 1024] if f >= hop_length])

        # convolutional hyperparameters (and transformer heads)
        # latent_dim % num_heads == 0
        conv_dim = trial.suggest_categorical("conv_dim", [64, 128, 256])
        num_conv = trial.suggest_int("num_conv", 2, 6)
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        dim_multiplier = trial.suggest_int("dim_multiplier", 2, 6)
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
        conv_dropout = trial.suggest_float("conv_dropout", 0.0, 0.5, step=0.05)

        latent_dim = num_heads * dim_multiplier * 8
        trial.set_user_attr("latent_dim", latent_dim)
        receptive_field = 1 + (kernel_size - 1) * num_conv
        trial.set_user_attr("receptive_field", receptive_field)

        # transformer hyperparameters
        # feedforward dim must be a multiple of the latent dim
        tf_layers = trial.suggest_int("tf_layers", 1, 4)
        ff_multiplier = trial.suggest_categorical("ff_multiplier", [1, 2, 4])
        transformer_dropout = trial.suggest_float("tf_dropout", 0.1, 0.5, step=0.05)

        tf_dim_ff = latent_dim * ff_multiplier
        trial.set_user_attr("tf_dim_ff", tf_dim_ff)

        # phoneme classifier
        phoneme_dropout = trial.suggest_float("phoneme_dropout", 0.1, 0.5, step=0.05)

        # other hyperparameters
        thresh = trial.suggest_float("thresh", 0.3, 0.7, step=0.05)
        muon_lr = trial.suggest_float("muon_lr", 1e-4, 1e-2, log=True)
        adam_lr = trial.suggest_float("adam_lr", 1e-4, 1e-2, log=True)
        pos_weight = trial.suggest_float("pos_weight", -2.0, 2.0)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        warmup_steps = trial.suggest_int("warmup_steps", 0, 500, step=100)
        b_loss_weight = trial.suggest_float("b_loss_weight", 0.0, 2.0, step=0.05)
        pf_loss_weight = trial.suggest_float("pf_loss_weight", 0.0, 2.0, step=0.05)
        ps_loss_weight = trial.suggest_float("ps_loss_weight", 0.0, 2.0, step=0.05)

        loss_weights = (b_loss_weight, pf_loss_weight, ps_loss_weight)

        temp_dir = tempfile.TemporaryDirectory()
        temp_bin_dir = Path(temp_dir.name) / "bin"

        hparams = dict(
            sr=sr,
            mels=mels,
            hop_length=hop_length,
            n_fft=n_fft,
            conv_dim=conv_dim,
            num_conv=num_conv,
            num_heads=num_heads,
            kernel_size=kernel_size,
            conv_dropout=conv_dropout,
            latent_dim=latent_dim,
            receptive_field=receptive_field,
            tf_layers=tf_layers,
            transformer_dropout=transformer_dropout,
            tf_dim_ff=tf_dim_ff,
            thresh=thresh,
            muon_lr=muon_lr,
            adam_lr=adam_lr,
            pos_weight=pos_weight,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            phoneme_dropout=phoneme_dropout,
            loss_weights=loss_weights,
        )

        proc = Preprocessor(argparse.Namespace(**{
            "dataset": data_dir,
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

        data_module = MinaDataModule(temp_bin_dir, batch_size, workers)
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
            sch_frequency=5,
            loss_weights=loss_weights
        )

        early_stop_callback = EarlyStopping(
            monitor="val/total_loss",
            patience=10,
            mode='min',
            verbose=True
        )

        pruning_callback = PyTorchLightningPruningCallback(
            trial=trial,
            monitor="val/total_loss",
        )

        trainer = lightning.Trainer(
            accelerator="auto",
            devices="auto",
            callbacks=[pruning_callback, early_stop_callback],
            logger=True,
            gradient_clip_val=1.0,
            accumulate_grad_batches=1,
            log_every_n_steps=10,
            check_val_every_n_epoch=5,
            precision='16-mixed'
        )

        trainer.logger.log_hyperparams(hparams)
        trainer.fit(model, data_module)
        temp_dir.cleanup()

        return trainer.callback_metrics["val/total_loss"].item()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("Pruning OOM trial")
            torch.cuda.empty_cache()
            gc.collect()
            del model
            raise optuna.exceptions.TrialPruned()
        else:
            raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--trials', type=int, default=500)
    args = parser.parse_args()

    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(),
        storage="sqlite:///db.sqlite3",
        study_name="mina"
    )

    study.optimize(
        lambda trial: objective(trial, args.data_dir, args.batch_size, args.workers),
        n_trials=args.trials,
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))