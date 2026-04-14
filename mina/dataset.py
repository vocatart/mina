import json
from pathlib import Path

import lightning
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


class MinaDataset(Dataset):
    def __init__(self, bin_dir: Path):
        self.bin_data = sorted(list(bin_dir.glob("**/*.npz")))
        self.bin_meta = json.load(open(bin_dir / "meta.json"))

        self.sr = self.bin_meta["hparams"]["sr"]
        self.n_mels = self.bin_meta["hparams"]["mels"]
        self.hop_length = self.bin_meta["hparams"]["hop_length"]
        self.n_fft = self.bin_meta["hparams"]["n_fft"]


    def __len__(self):
        return len(self.bin_data)

    def __getitem__(self, idx):
        data = np.load(str(self.bin_data[idx]))

        return {
            "mel": torch.FloatTensor(data["mel"]),
            "boundaries": torch.LongTensor(data["bounds"]),
            "phonemes": torch.LongTensor(data["phonemes"]),
        }

    @staticmethod
    def collate_fn(batch):
        max_len = max(item["mel"].size(0) for item in batch)

        mels, bounds, phonemes, lengths = list(), list(), list(), list()

        # pad each item to longest sequence in batch
        # retain original lengths for masking
        for item in batch:
            orig_len = item["mel"].size(0)
            n_mels = item["mel"].size(1)

            mel_pad = torch.zeros(max_len, n_mels)
            mel_pad[:orig_len] = item["mel"]
            mels.append(mel_pad)

            bound_pad = torch.zeros(max_len, dtype=torch.long)
            bound_pad[:orig_len] = item["boundaries"]
            bounds.append(bound_pad)

            phoneme_pad = torch.zeros(max_len, dtype=torch.long)
            phoneme_pad[:orig_len] = item["phonemes"]
            phonemes.append(phoneme_pad)

            lengths.append(orig_len)

        # mel: (B, max_len, n_mels)
        # boundaries: (B, max_len)
        # phonemes: (B, max_len)
        # lengths: (B,)
        return {
            "mel": torch.stack(mels, dim=0),
            "boundaries": torch.stack(bounds, dim=0),
            "phonemes": torch.stack(phonemes, dim=0),
            "lengths": torch.tensor(lengths, dtype=torch.long),
        }

class MinaDataModule(lightning.LightningDataModule):
    def __init__(self, bin_dir: Path, batch_size: int, n_workers: int):
        super().__init__()
        self.train, self.val, self.test = None, None, None

        self.bin_dir = bin_dir
        self.bin_meta = json.load(open(bin_dir / "meta.json"))
        self.batch_size = batch_size
        self.n_workers = n_workers

        self.sr = self.bin_meta["hparams"]["sr"]
        self.n_mels = self.bin_meta["hparams"]["mels"]
        self.hop_length = self.bin_meta["hparams"]["hop_length"]
        self.n_fft = self.bin_meta["hparams"]["n_fft"]
        self.valid_split = self.bin_meta["hparams"]["valid_split"]
        self.rec_max_len = self.bin_meta["hparams"]["max_len"]

        self.persist = True if n_workers > 0 else False

    def setup(self, stage=None):
        dataset = MinaDataset(bin_dir=self.bin_dir)

        total_size = len(dataset)
        val_size = int(total_size * self.valid_split)
        test_size = int(total_size * self.valid_split)
        train_size = total_size - val_size - test_size

        self.train, self.val, self.test = torch.utils.data.random_split(
            dataset,
    [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(76_805)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            collate_fn=MinaDataset.collate_fn,
            persistent_workers=self.persist,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            collate_fn=MinaDataset.collate_fn,
            persistent_workers=self.persist,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            collate_fn=MinaDataset.collate_fn,
            persistent_workers=self.persist,
        )