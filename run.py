import argparse
from pathlib import Path

import librosa
import numpy as np
import onnxruntime
import textgrid
import yaml
from librosa import feature


def get_mel(audio: Path, sr: int, n_mels: int, hop_length: int, n_fft: int) -> np.ndarray:
    y, _ = librosa.load(audio, sr=sr)
    mel = feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
    )

    return librosa.power_to_db(mel, ref=np.max).T


def make_textgrid(phonemes: np.ndarray, boundaries: np.ndarray, hop_length: int, sr: int, phoneme_map: dict) -> textgrid.TextGrid:
    max_len = len(phonemes)
    duration = max_len * hop_length / sr

    tg = textgrid.TextGrid(minTime=0, maxTime=duration)
    tier = textgrid.IntervalTier(name="phones", minTime=0, maxTime=duration)

    bound_frames = np.where(boundaries == 1)[0].tolist()
    seg_starts = [0] + [b for b in bound_frames if b > 0]
    seg_ends = seg_starts[1:] + [max_len]

    for start_f, end_f in zip(seg_starts, seg_ends):
        if start_f >= end_f:
            continue

        start_t = start_f * hop_length / sr
        end_t = end_f * hop_length / sr

        phone_idx = int(np.bincount(phonemes[start_f:end_f]).argmax())
        label = phoneme_map.get(phone_idx, phoneme_map.get(str(phone_idx), ""))

        tier.add(start_t, end_t, label)

    tg.append(tier)
    return tg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("hparams")
    parser.add_argument("audio")
    parser.add_argument("output")

    args = parser.parse_args()

    print(f"Loading hyperparameters: {args.hparams}")
    with open(args.hparams) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    sr = hparams["sr"]
    n_mels = hparams["d_mel"]
    hop_length = hparams["hop_length"]
    n_fft = hparams["n_fft"]
    print(f"Constructing mel-spectrogram: sr: {sr}, n_mels: {n_mels}, hop_length: {hop_length}, n_fft: {n_fft}")
    mel = get_mel(args.audio, sr, n_mels, hop_length, n_fft)

    print(f"Loading model {args.model}")
    session = onnxruntime.InferenceSession(args.model)

    print(f"Running inference on {args.audio}")
    preds = session.run(["phonemes", "boundaries"], {"mel": mel[np.newaxis, ...]})
    phonemes, boundaries = preds[0][0], preds[1][0]

    phoneme_map = {int(k): v for k, v in hparams["phoneme_map"].items()}
    output_path = Path(args.output) / (Path(args.audio).stem + ".TextGrid")
    print(f"Constructing TextGrid: {output_path}")
    grid = make_textgrid(phonemes, boundaries, hop_length, sr, phoneme_map)
    grid.write(str(output_path))