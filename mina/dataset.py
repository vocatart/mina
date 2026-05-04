import json
from pathlib import Path

import lightning
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

class MinaDataset(Dataset):
    def __init__(self, bin_dir: Path) -> None:
        self.bin_data = sorted(list(bin_dir.glob("**/*.npz")))
        self.bin_meta = json.load(open(bin_dir / "meta.json"))

        self.sr = self.bin_meta["hparams"]["sr"]
        self.n_mels = self.bin_meta["hparams"]["mels"]
        self.hop_length = self.bin_meta["hparams"]["hop_length"]
        self.n_fft = self.bin_meta["hparams"]["n_fft"]

    def __len__(self) -> int:
        return len(self.bin_data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = np.load(str(self.bin_data[idx]))

        return {
            "mel": torch.tensor(data["mel"], dtype=torch.float32),
            "boundaries": torch.tensor(data["bounds"], dtype=torch.long),
            "frame_phonemes": torch.tensor(data["frame_phonemes"], dtype=torch.long),
            "segment_phonemes": torch.tensor(data["segment_phonemes"], dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        Collate function for mina dataset. Pads to the longest audio sequence and batch and retains original lengths for masking.

        Args:
            batch (list[dict[str, torch.Tensor]]): A list of dictionaries containing mels, boundaries, and phonemes

        Returns:
            A dictionary of mels, boundaries, phonemes, and lengths.
        """
        max_len = max(item["mel"].size(0) for item in batch)
        max_seg_len = max(item["segment_phonemes"].size(0) for item in batch)
        mels, bounds, frame_phonemes, segment_phonemes, frame_lengths, segment_lengths = list(), list(), list(), list(), list(), list()

        # pad each item to longest sequence in batch
        # retain original lengths for masking
        for item in batch:
            orig_len = item["mel"].size(0)
            orig_seg_len = item["segment_phonemes"].size(0)
            n_mels = item["mel"].size(1)

            mel_pad = torch.zeros(max_len, n_mels)
            mel_pad[:orig_len] = item["mel"]
            mels.append(mel_pad)

            bound_pad = torch.zeros(max_len, dtype=torch.long)
            bound_pad[:orig_len] = item["boundaries"]
            bounds.append(bound_pad)

            frame_phoneme_pad = torch.zeros(max_len, dtype=torch.long)
            frame_phoneme_pad[:orig_len] = item["frame_phonemes"]
            frame_phonemes.append(frame_phoneme_pad)

            segment_phoneme_pad = torch.zeros(max_seg_len, dtype=torch.long)
            segment_phoneme_pad[:orig_seg_len] = item["segment_phonemes"]
            segment_phonemes.append(segment_phoneme_pad)

            frame_lengths.append(orig_len)
            segment_lengths.append(orig_seg_len)

        # mel: (B, max_len, n_mels)
        # boundaries: (B, max_len)
        # frame_phonemes: (B, max_len)
        # segment_phoemes: (B, max_seg_len)
        # frame_lengths: (B,)
        # segment_lenghts: (B,)
        return {
            "mel": torch.stack(mels, dim=0),
            "boundaries": torch.stack(bounds, dim=0),
            "frame_phonemes": torch.stack(frame_phonemes, dim=0),
            "segment_phonemes": torch.stack(segment_phonemes, dim=0),
            "frame_lengths": torch.tensor(frame_lengths, dtype=torch.long),
            "segment_lengths": torch.tensor(segment_lengths, dtype=torch.long),
        }

class MinaDataModule(lightning.LightningDataModule):
    def __init__(self, bin_dir: Path, batch_size: int, n_workers: int) -> None:
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
        self.vocab_size = self.bin_meta["hparams"]["vocab_size"]
        self.phoneme_map = {int(k): v for k, v in self.bin_meta["phoneme_map"].items()}

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
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            collate_fn=MinaDataset.collate_fn,
            persistent_workers=True,
        )