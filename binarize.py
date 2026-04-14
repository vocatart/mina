import argparse
import datetime
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import textgrid
from librosa import feature
from tqdm import tqdm


class Preprocessor:
    def __init__(self, a):
        self.db = Path(a.dataset)
        self.out = Path(a.output)

        self.sr = a.sr
        self.mels = a.mels
        self.hop_length = a.hop_length
        self.n_fft = a.n_fft

        self.valid_split = a.val_split
        self.time_split = a.time_split
        self.audio_types = a.audio_types
        self.workers = a.workers

        self.out.mkdir(parents=True, exist_ok=True)

        self.longest_seen_sequence = 0
        self.total_length = 0.0

        self.audio_files = list()
        for ext in self.audio_types:
            matches = self.db.glob(f"**/*.{ext}")

            for audio_file in matches:
                # ignore audio without labels
                if audio_file.with_suffix(".TextGrid").exists():
                    self.audio_files.append(audio_file)

        phoneme_idx = list()
        for audio_file in self.audio_files:
            tg = textgrid.TextGrid.fromFile(audio_file.with_suffix(".TextGrid"))
            tier = tg[0]
            for interval in tier.intervals:
                if len(interval.mark) > 5:
                    print(audio_file, interval.mark, tier.name)
                phoneme_idx.append(interval.mark)

        self.phoneme_map = sorted(tuple(set(phoneme_idx)))

        print(f"Found phonemes: {self.phoneme_map}")

    def update_max_seq_len(self, seq_len: int):
        if seq_len > self.longest_seen_sequence:
            # print(f"New longest sequence {seq_len}")
            self.longest_seen_sequence = seq_len

    def get_dur(self, audio_file: Path):
        dur = librosa.get_duration(path=audio_file)
        self.total_length += dur
        return dur

    def process_audio(self):
        total_length = 0.0
        longest_seen_sequence = 0

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self.process_single_audio, audio_file) for audio_file in self.audio_files}

            for future in tqdm(as_completed(futures), total=len(self.audio_files)):
                audio_length, max_seq_len = future.result()
                total_length += audio_length
                longest_seen_sequence = max(longest_seen_sequence, max_seq_len)

        self.total_length = total_length
        self.longest_seen_sequence = longest_seen_sequence

    def process_single_audio(self, audio_file: Path):
        audio_length = self.get_dur(audio_file)
        current_pos = 0.0
        slice_idx = 0
        max_seq_len = 0

        tg = textgrid.TextGrid.fromFile(str(audio_file.with_suffix(".TextGrid")))

        while current_pos < audio_length:
            closest_time = self.snap_to_next_interval(tg, current_pos, audio_length)

            y, _ = librosa.load(audio_file, sr=self.sr, offset=current_pos, duration=(closest_time - current_pos))

            min_duration = self.n_fft / self.sr
            if closest_time - current_pos < min_duration:
                current_pos = closest_time
                slice_idx += 1
                continue

            mel_dbs = self.get_mel(y)

            seq_len = len(mel_dbs)
            max_seq_len = max(max_seq_len, seq_len)

            boundaries, phonemes = self.get_boundaries_and_phonemes(tg, seq_len, current_pos)

            with open(os.path.join(self.out, os.path.basename(audio_file) + str(slice_idx) + ".npz"), "wb") as f:
                np.savez(f, mel=mel_dbs, bounds=boundaries, phonemes=phonemes)

            current_pos = closest_time
            slice_idx += 1

        return audio_length, max_seq_len

    def snap_to_next_interval(self, tg: textgrid.TextGrid, position: float, duration: float):
        # TODO: assumes phonemes are on tier 0 (they usually are)
        tier = tg.tiers[0]
        target_time = position + self.time_split

        for interval in tier.intervals:
            if interval.minTime >= target_time:
                return interval.minTime

        return duration

    def get_mel(self, audio):
        mel = feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )

        return librosa.power_to_db(mel, ref=np.max).T

    def get_boundaries_and_phonemes(self, tg: textgrid.TextGrid, seq_len: int, offset: float):
        tier = tg.tiers[0]
        boundaries = np.zeros(seq_len, dtype=np.int64)
        phonemes = np.zeros(seq_len, dtype=np.int64)

        for interval in tier.intervals:
            start_idx = int((interval.minTime - offset) * self.sr / self.hop_length)
            end_idx = int((interval.maxTime - offset) * self.sr / self.hop_length)

            start_idx = max(0, min(seq_len, start_idx))
            end_idx = max(0, min(seq_len, end_idx))

            if 0 <= start_idx < seq_len:
                boundaries[start_idx] = 1

            if start_idx < end_idx:
                phoneme_id = self.phoneme_map.index(interval.mark)
                phonemes[start_idx:end_idx + 1] = phoneme_id

        return boundaries, phonemes

    def save_metadata(self):
        phone_dict = {index: phone for index, phone in enumerate(self.phoneme_map)}

        json_dict = {
            "phoneme_map": phone_dict,
            "hparams": {
                "sr": self.sr,
                "mels": self.mels,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "valid_split": self.valid_split,
                "max_len": self.longest_seen_sequence
            }
        }

        json.dump(json_dict, open(os.path.join(self.out, "meta.json"), "w"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output")

    parser.add_argument("--sr", type=int, default=48000, help="Target sample rate")
    parser.add_argument("--mels", type=int, default=80, help="Number of mel bins")
    parser.add_argument("--hop_length", type=int, default=160, help="Hop length")
    parser.add_argument("--n_fft", type=int, default=1024, help="FFT length")

    parser.add_argument("--time_split", type=int, default=10, help="Audio segment length")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--audio_types", nargs="+", type=str, help="Audio types to look for")
    parser.add_argument("--workers", type=int, default=16, help="Number of workers")

    proc = Preprocessor(parser.parse_args())

    proc.process_audio()
    proc.save_metadata()

    print(f"Recommended positional encoding length: {proc.longest_seen_sequence}")
    print(f"Total dataset length: {str(datetime.timedelta(seconds=proc.total_length))}")