import os
import glob
import math
import json
import random
import argparse
from dataclasses import dataclass

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class ChunkItem:
    file_path: str
    start_sec: float
    label: int


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_wav_files(folder: str):
    patterns = ["*.wav", "*.WAV"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, "**", p), recursive=True))
    return sorted(files)


def build_chunk_index(
    files,
    label,
    segment_seconds,
    hop_seconds,
    max_chunks=None,
):
    chunks = []
    for fp in files:
        try:
            duration = librosa.get_duration(path=fp)
        except Exception as exc:
            print(f"[WARN] Impossibile leggere durata: {fp} -> {exc}")
            continue

        if duration < segment_seconds:
            continue

        n = int(math.floor((duration - segment_seconds) / hop_seconds)) + 1
        for i in range(n):
            start = i * hop_seconds
            chunks.append(ChunkItem(file_path=fp, start_sec=start, label=label))

    if max_chunks is not None and len(chunks) > max_chunks:
        random.shuffle(chunks)
        chunks = chunks[:max_chunks]

    return chunks


def train_val_split(items, val_ratio=0.15):
    random.shuffle(items)
    n_val = int(len(items) * val_ratio)
    val = items[:n_val]
    train = items[n_val:]
    return train, val


class GenreChunkDataset(Dataset):
    def __init__(
        self,
        items,
        sr=22050,
        segment_seconds=5.0,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        fmin=30,
        fmax=11025,
    ):
        self.items = items
        self.sr = sr
        self.segment_seconds = segment_seconds
        self.segment_samples = int(sr * segment_seconds)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

    def __len__(self):
        return len(self.items)

    def _load_audio_chunk(self, file_path, start_sec):
        y, _ = librosa.load(
            file_path,
            sr=self.sr,
            mono=True,
            offset=float(start_sec),
            duration=self.segment_seconds,
        )

        if y.shape[0] < self.segment_samples:
            pad = self.segment_samples - y.shape[0]
            y = np.pad(y, (0, pad), mode="constant")
        elif y.shape[0] > self.segment_samples:
            y = y[: self.segment_samples]

        return y

    def _to_logmel(self, y):
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)

        # Normalizzazione in [0, 1] per stabilita training
        mel_norm = (mel_db + 80.0) / 80.0
        mel_norm = np.clip(mel_norm, 0.0, 1.0)
        return mel_norm.astype(np.float32)

    def __getitem__(self, idx):
        item = self.items[idx]
        y = self._load_audio_chunk(item.file_path, item.start_sec)
        mel = self._to_logmel(y)

        # (1, n_mels, time)
        x = np.expand_dims(mel, axis=0)
        y_label = np.float32(item.label)

        return torch.from_numpy(x), torch.tensor(y_label)


class SmallGenreCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        logits = self.classifier(x).squeeze(1)
        return logits


def run_epoch(model, loader, criterion, optimizer, device):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        total_loss += loss.item() * x.size(0)
        correct += (preds == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser(description="Training classificatore di genere su 2 domini WAV")
    parser.add_argument("--domain_a_dir", type=str, default=r"C:\CycleGan\STFT\Audio_dominio_A_v2")
    parser.add_argument("--domain_b_dir", type=str, default=r"C:\CycleGan\STFT\Audio_dominio_B_v2")
    parser.add_argument("--output_dir", type=str, default=r"C:\CycleGan\STFT\genre_classifier")

    parser.add_argument("--segment_seconds", type=float, default=5.0)
    parser.add_argument("--hop_seconds", type=float, default=2.5)
    parser.add_argument("--max_chunks_per_class", type=int, default=12000)
    parser.add_argument("--val_ratio", type=float, default=0.15)

    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--n_mels", type=int, default=128)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    files_a = list_wav_files(args.domain_a_dir)
    files_b = list_wav_files(args.domain_b_dir)

    if not files_a:
        raise FileNotFoundError(f"Nessun WAV trovato in {args.domain_a_dir}")
    if not files_b:
        raise FileNotFoundError(f"Nessun WAV trovato in {args.domain_b_dir}")

    print(f"File dominio A: {len(files_a)}")
    print(f"File dominio B: {len(files_b)}")

    items_a = build_chunk_index(
        files_a,
        label=0,
        segment_seconds=args.segment_seconds,
        hop_seconds=args.hop_seconds,
        max_chunks=args.max_chunks_per_class,
    )
    items_b = build_chunk_index(
        files_b,
        label=1,
        segment_seconds=args.segment_seconds,
        hop_seconds=args.hop_seconds,
        max_chunks=args.max_chunks_per_class,
    )

    if not items_a or not items_b:
        raise RuntimeError("Chunk insufficienti: controlla durata audio o parametri segment/hop.")

    # Bilanciamento classi
    m = min(len(items_a), len(items_b))
    random.shuffle(items_a)
    random.shuffle(items_b)
    items = items_a[:m] + items_b[:m]

    train_items, val_items = train_val_split(items, val_ratio=args.val_ratio)

    print(f"Chunk train: {len(train_items)}")
    print(f"Chunk val:   {len(val_items)}")

    ds_train = GenreChunkDataset(
        train_items,
        sr=args.sr,
        segment_seconds=args.segment_seconds,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmax=args.sr // 2,
    )
    ds_val = GenreChunkDataset(
        val_items,
        sr=args.sr,
        segment_seconds=args.segment_seconds,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmax=args.sr // 2,
    )

    loader_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = SmallGenreCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_path = os.path.join(args.output_dir, "genre_classifier_best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, loader_train, criterion, optimizer, device)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(model, loader_val, criterion, None, device)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "sr": args.sr,
                        "segment_seconds": args.segment_seconds,
                        "n_fft": args.n_fft,
                        "hop_length": args.hop_length,
                        "n_mels": args.n_mels,
                    },
                    "label_map": {"0": "domain_a", "1": "domain_b"},
                    "best_val_acc": best_val_acc,
                },
                best_path,
            )
            print(f"Nuovo best model salvato: {best_path}")

    meta_path = os.path.join(args.output_dir, "training_summary.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "domain_a_dir": args.domain_a_dir,
                "domain_b_dir": args.domain_b_dir,
                "num_files_a": len(files_a),
                "num_files_b": len(files_b),
                "train_chunks": len(train_items),
                "val_chunks": len(val_items),
                "best_val_acc": best_val_acc,
                "model_path": best_path,
            },
            f,
            indent=2,
        )

    print("Training completato.")
    print(f"Best model: {best_path}")
    print(f"Summary:    {meta_path}")


if __name__ == "__main__":
    main()
