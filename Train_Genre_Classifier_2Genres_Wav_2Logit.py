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


def list_audio_files(folder: str):
    patterns = ["*.wav", "*.WAV", "*.mp3", "*.MP3", "*.opus", "*.OPUS"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, "**", p), recursive=True))
    return sorted(files)


def is_audio_readable(file_path: str, sr: int, probe_seconds: float = 0.25):
    """Ritorna True se il file e decodificabile almeno su una porzione iniziale."""
    try:
        y, _ = librosa.load(
            file_path,
            sr=sr,
            mono=True,
            offset=0.0,
            duration=probe_seconds,
        )
        return y is not None and y.size > 0
    except Exception:
        return False


def build_chunk_index(files, label, segment_seconds, hop_seconds, max_chunks=None, sr=22050):
    items = []
    for fp in files:
        if not is_audio_readable(fp, sr=sr):
            print(f"[WARN] File audio non decodificabile, salto: {fp}")
            continue

        try:
            duration = librosa.get_duration(path=fp)
        except Exception as exc:
            print(f"[WARN] Durata non leggibile: {fp} -> {exc}")
            continue

        if duration < segment_seconds:
            continue

        n = int(math.floor((duration - segment_seconds) / hop_seconds)) + 1
        for i in range(n):
            items.append(ChunkItem(fp, i * hop_seconds, label))

    if max_chunks is not None and len(items) > max_chunks:
        random.shuffle(items)
        items = items[:max_chunks]

    return items


def train_val_split(items, val_ratio=0.15):
    random.shuffle(items)
    n_val = int(len(items) * val_ratio)
    return items[n_val:], items[:n_val]


class RawWavChunkDataset(Dataset):
    def __init__(
        self,
        items,
        sr=22050,
        segment_seconds=5.0,
        pre_emphasis=0.0,
    ):
        self.items = items
        self.sr = sr
        self.segment_seconds = segment_seconds
        self.segment_samples = int(sr * segment_seconds)
        self.pre_emphasis = pre_emphasis
        self.unreadable_files = set()

    def __len__(self):
        return len(self.items)

    def _load_chunk(self, file_path, start_sec):
        if file_path in self.unreadable_files:
            return np.zeros(self.segment_samples, dtype=np.float32)

        try:
            y, _ = librosa.load(
                file_path,
                sr=self.sr,
                mono=True,
                offset=float(start_sec),
                duration=self.segment_seconds,
            )
        except Exception as exc:
            self.unreadable_files.add(file_path)
            print(f"[WARN] Errore decodifica chunk, uso silenzio e marco file come non leggibile: {file_path} -> {exc}")
            return np.zeros(self.segment_samples, dtype=np.float32)

        if y.shape[0] < self.segment_samples:
            y = np.pad(y, (0, self.segment_samples - y.shape[0]), mode="constant")
        elif y.shape[0] > self.segment_samples:
            y = y[: self.segment_samples]

        # opzionale: enfatizza alte frequenze, utile su alcuni task di timbro
        if self.pre_emphasis > 0.0:
            y = np.append(y[0], y[1:] - self.pre_emphasis * y[:-1])

        # Normalizzazione peak-safe per stabilita training
        peak = np.max(np.abs(y))
        if peak > 1e-9:
            y = y / peak

        return y.astype(np.float32)

    def __getitem__(self, idx):
        item = self.items[idx]
        wav = self._load_chunk(item.file_path, item.start_sec)

        # output shape: (T,)
        x = torch.from_numpy(wav)
        y = torch.tensor(item.label, dtype=torch.long)
        return x, y


class GenreWaveformCNN(nn.Module):
    """
    Classificatore binario waveform-based a 2 logit (CrossEntropyLoss).
    Input:
      - (B, T) oppure (B, 1, T)
    Output:
      - logits shape (B, 2)
        logits[:, 0] -> dominio A
        logits[:, 1] -> dominio B

    A inferenza:
        probs = torch.softmax(logits, dim=1)
        p_A = probs[:, 0]
        p_B = probs[:, 1]
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(256, 2)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.net(x)
        x = x.squeeze(-1)
        logits = self.head(x)  # shape (B, 2)
        return logits


def run_epoch(model, loader, criterion, optimizer, device):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.long().to(device, non_blocking=True)

        logits = model(x)  # (B, 2)
        loss = criterion(logits, y)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * x.size(0)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Training classificatore genere (raw WAV, 2 logit) su 2 domini")

    parser.add_argument("--domain_a_dir", type=str, default=r"C:\CycleGan\STFT\Audio_Dataset_Clean_Wav\Audio_Dataset_Piano")
    parser.add_argument("--domain_b_dir", type=str, default=r"C:\CycleGan\STFT\Audio_Dataset_Clean_Wav\Audio_Dataset_Violin")
    parser.add_argument("--output_dir", type=str, default=r"C:\CycleGan\STFT\genre_classifier_wav")

    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--segment_seconds", type=float, default=3.0)
    parser.add_argument("--hop_seconds", type=float, default=1.5)
    parser.add_argument("--pre_emphasis", type=float, default=0.0)

    parser.add_argument("--max_chunks_per_class", type=int, default=20000)
    parser.add_argument("--val_ratio", type=float, default=0.15)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    files_a = list_audio_files(args.domain_a_dir)
    files_b = list_audio_files(args.domain_b_dir)

    if not files_a:
        raise FileNotFoundError(f"Nessun file audio trovato in {args.domain_a_dir} (estensioni supportate: wav, mp3, opus)")
    if not files_b:
        raise FileNotFoundError(f"Nessun file audio trovato in {args.domain_b_dir} (estensioni supportate: wav, mp3, opus)")

    print(f"File dominio A: {len(files_a)}")
    print(f"File dominio B: {len(files_b)}")

    items_a = build_chunk_index(
        files_a,
        label=0,
        segment_seconds=args.segment_seconds,
        hop_seconds=args.hop_seconds,
        max_chunks=args.max_chunks_per_class,
        sr=args.sr,
    )
    items_b = build_chunk_index(
        files_b,
        label=1,
        segment_seconds=args.segment_seconds,
        hop_seconds=args.hop_seconds,
        max_chunks=args.max_chunks_per_class,
        sr=args.sr,
    )

    if not items_a or not items_b:
        raise RuntimeError("Chunk insufficienti: controlla parametri segment/hop.")

    # Bilanciamento classi
    m = min(len(items_a), len(items_b))
    random.shuffle(items_a)
    random.shuffle(items_b)
    items = items_a[:m] + items_b[:m]

    train_items, val_items = train_val_split(items, val_ratio=args.val_ratio)

    print(f"Chunk train: {len(train_items)}")
    print(f"Chunk val:   {len(val_items)}")

    ds_train = RawWavChunkDataset(
        train_items,
        sr=args.sr,
        segment_seconds=args.segment_seconds,
        pre_emphasis=args.pre_emphasis,
    )
    ds_val = RawWavChunkDataset(
        val_items,
        sr=args.sr,
        segment_seconds=args.segment_seconds,
        pre_emphasis=args.pre_emphasis,
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

    model = GenreWaveformCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_ckpt_path = os.path.join(args.output_dir, "genre_classifier_wav2logit_best_ckpt.pt")
    best_jit_path = os.path.join(args.output_dir, "genre_classifier_wav2logit_best_jit.pt")

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
                    "model_class": "GenreWaveformCNN_2Logit",
                    "config": {
                        "sr": args.sr,
                        "segment_seconds": args.segment_seconds,
                        "hop_seconds": args.hop_seconds,
                        "pre_emphasis": args.pre_emphasis,
                    },
                    "label_map": {"0": "domain_a", "1": "domain_b"},
                    "best_val_acc": best_val_acc,
                },
                best_ckpt_path,
            )

            model.eval()
            scripted = torch.jit.script(model.cpu())
            scripted.save(best_jit_path)
            model.to(device)

            print(f"Nuovo best model: {best_ckpt_path}")
            print(f"TorchScript: {best_jit_path}")

    summary_path = os.path.join(args.output_dir, "training_summary_wav2logit.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "domain_a_dir": args.domain_a_dir,
                "domain_b_dir": args.domain_b_dir,
                "num_files_a": len(files_a),
                "num_files_b": len(files_b),
                "train_chunks": len(train_items),
                "val_chunks": len(val_items),
                "best_val_acc": best_val_acc,
                "best_ckpt_path": best_ckpt_path,
                "best_jit_path": best_jit_path,
            },
            f,
            indent=2,
        )

    print("Training completato.")
    print(f"Best CKPT: {best_ckpt_path}")
    print(f"Best JIT:  {best_jit_path}")
    print(f"Summary:   {summary_path}")


if __name__ == "__main__":
    main()
