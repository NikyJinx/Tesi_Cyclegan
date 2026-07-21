import os
from typing import List, Sequence, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch

from CycleGan_v6 import Generator_2_1_2D


SR = 22050
N_FFT = 1024
HOP_LENGTH = 256
TARGET_FRAMES = 2048
OVERLAP_RATIO = 0.5
TOP_DB = 80.0
PEAK_HEADROOM = 0.95
SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".opus", ".flac")

# =========================
# CONFIGURAZIONE RUN DIRETTA
# =========================
INPUT_DIR = "C:\\CycleGan\\STFT\\Input"
CHECKPOINT_PATH = "C:\\CycleGan\\STFT\\checkpoints\\checkpoint_epoch_093.pt"
OUTPUT_AUDIO_DIR = "C:\\CycleGan\\STFT\\Output"
GENERATOR_KEY = "G_AB"  # "G_AB" oppure "G_BA"

BATCH_SIZE = 1
GRIFFIN_LIM_ITERS = 64

RUN_SR = SR
RUN_N_FFT = N_FFT
RUN_HOP_LENGTH = HOP_LENGTH
RUN_TARGET_FRAMES = TARGET_FRAMES
RUN_OVERLAP_RATIO = OVERLAP_RATIO


def compute_starts(total_frames: int, target_frames: int, stride_frames: int) -> List[int]:
    if total_frames <= target_frames:
        return [0]

    starts = list(range(0, total_frames - target_frames + 1, stride_frames))
    last_start = total_frames - target_frames
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def pad_block(block: np.ndarray, pad_value: float, target_frames: int) -> np.ndarray:
    pad_width = target_frames - block.shape[1]
    if pad_width <= 0:
        return block
    return np.pad(block, ((0, 0), (0, pad_width)), constant_values=pad_value)


def build_input_blocks(
    file_path: str,
    sr: int,
    n_fft: int,
    hop_length: int,
    target_frames: int,
    overlap_ratio: float,
) -> Tuple[np.ndarray, List[int], int, int]:
    stride_frames = int(target_frames * (1.0 - overlap_ratio))
    if stride_frames <= 0:
        raise ValueError("overlap_ratio non valido: stride <= 0")

    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    stft_matrix = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    stft_matrix = stft_matrix[:-1, :]

    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    mag_db = librosa.amplitude_to_db(magnitude, ref=np.max, top_db=TOP_DB)
    mag_norm = np.clip((mag_db / 40.0) + 1.0, -1.0, 1.0)

    sin_phase = np.sin(phase).astype(np.float32)
    cos_phase = np.cos(phase).astype(np.float32)

    total_frames = mag_norm.shape[1]
    starts = compute_starts(total_frames, target_frames, stride_frames)

    blocks = []
    for start in starts:
        end = min(start + target_frames, total_frames)

        mag_block = pad_block(mag_norm[:, start:end], -1.0, target_frames)
        sin_block = pad_block(sin_phase[:, start:end], 0.0, target_frames)
        cos_block = pad_block(cos_phase[:, start:end], 1.0, target_frames)

        block = np.stack((mag_block, sin_block, cos_block), axis=0).astype(np.float32)
        blocks.append(block)

    return np.stack(blocks, axis=0), starts, len(audio), total_frames


def load_generator(checkpoint_path: str, generator_key: str, device: torch.device) -> Generator_2_1_2D:
    generator = Generator_2_1_2D().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if generator_key not in checkpoint:
        available_keys = ", ".join(sorted(checkpoint.keys()))
        raise KeyError(
            f"Chiave '{generator_key}' non trovata nel checkpoint. Chiavi disponibili: {available_keys}"
        )

    generator.load_state_dict(checkpoint[generator_key])
    generator.eval()
    return generator


def run_generator(
    generator: Generator_2_1_2D,
    blocks: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs = []

    with torch.no_grad():
        for start_idx in range(0, len(blocks), batch_size):
            batch = torch.from_numpy(blocks[start_idx:start_idx + batch_size]).to(device)
            pred = generator(batch).cpu().numpy().astype(np.float32)
            outputs.append(pred)

    return np.concatenate(outputs, axis=0)


def merge_magnitude_blocks(
    predicted_blocks: np.ndarray,
    starts: Sequence[int],
    total_frames: int,
    target_frames: int,
) -> np.ndarray:
    merged_mag = np.zeros((512, total_frames), dtype=np.float32)
    weights = np.zeros(total_frames, dtype=np.float32)

    for block_idx, start in enumerate(starts):
        end = min(start + target_frames, total_frames)
        valid_width = end - start

        merged_mag[:, start:end] += predicted_blocks[block_idx, 0, :, :valid_width]
        weights[start:end] += 1.0

    weights = np.maximum(weights, 1e-8)
    return merged_mag / weights[np.newaxis, :]


def reconstruct_audio_griffin_lim(
    merged_mag_norm: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    target_length: int,
    griffin_lim_iters: int,
) -> np.ndarray:
    merged_mag_norm = np.clip(merged_mag_norm, -1.0, 1.0)
    mag_db = 40.0 * (merged_mag_norm - 1.0)
    magnitude = librosa.db_to_amplitude(mag_db)

    # Ripristina il bin di Nyquist: da 512 a 513 per n_fft=1024.
    if magnitude.shape[0] == (n_fft // 2):
        magnitude = np.pad(magnitude, ((0, 1), (0, 0)), mode="constant")

    if magnitude.shape[0] != (n_fft // 2 + 1):
        raise ValueError(
            f"Magnitudine incompatibile: {magnitude.shape[0]} righe con n_fft={n_fft}"
        )

    audio = librosa.griffinlim(
        S=magnitude,
        n_iter=griffin_lim_iters,
        hop_length=hop_length,
        win_length=n_fft,
        length=target_length,
    )

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = (audio / peak) * PEAK_HEADROOM

    return audio.astype(np.float32)


def infer_file(
    file_path: str,
    generator: Generator_2_1_2D,
    output_dir: str,
    device: torch.device,
    generator_key: str,
    batch_size: int,
    griffin_lim_iters: int,
    sr: int,
    n_fft: int,
    hop_length: int,
    target_frames: int,
    overlap_ratio: float,
) -> str:
    blocks, starts, audio_length, total_frames = build_input_blocks(
        file_path=file_path,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        target_frames=target_frames,
        overlap_ratio=overlap_ratio,
    )

    predictions = run_generator(generator, blocks, device=device, batch_size=batch_size)
    merged_mag_norm = merge_magnitude_blocks(predictions, starts, total_frames, target_frames)

    audio = reconstruct_audio_griffin_lim(
        merged_mag_norm=merged_mag_norm,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        target_length=audio_length,
        griffin_lim_iters=griffin_lim_iters,
    )

    input_name = os.path.splitext(os.path.basename(file_path))[0]
    out_name = f"{input_name}_{generator_key}_griffinlim.wav"
    out_path = os.path.join(output_dir, out_name)
    sf.write(out_path, audio, sr)
    return out_path


def inference_and_reconstruct_griffin_lim(
    input_dir: str = "C:\\CycleGan\\STFT\\Input",
    checkpoint_path: str = "C:\\CycleGan\\STFT\\checkpoints\\checkpoint_epoch_093.pt",
    output_audio_dir: str = "C:\\CycleGan\\STFT\\Output",
    generator_key: str = "G_AB",
    batch_size: int = 1,
    griffin_lim_iters: int = 64,
    sr: int = SR,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    target_frames: int = TARGET_FRAMES,
    overlap_ratio: float = OVERLAP_RATIO,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_audio_dir, exist_ok=True)

    audio_files = sorted(
        entry.path
        for entry in os.scandir(input_dir)
        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in SUPPORTED_EXTENSIONS
    )

    if not audio_files:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise RuntimeError(
            f"Nessun file audio trovato in '{input_dir}'. Estensioni supportate: {supported}"
        )

    generator = load_generator(checkpoint_path, generator_key, device)

    print(f"Device: {device}")
    print(f"Generatore caricato da: {checkpoint_path}")
    print(f"Chiave generatore: {generator_key}")
    print(f"File da processare: {len(audio_files)}")
    print(f"Ricostruzione fase: Griffin-Lim (iterazioni={griffin_lim_iters})")

    for index, file_path in enumerate(audio_files, start=1):
        out_path = infer_file(
            file_path=file_path,
            generator=generator,
            output_dir=output_audio_dir,
            device=device,
            generator_key=generator_key,
            batch_size=batch_size,
            griffin_lim_iters=griffin_lim_iters,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            target_frames=target_frames,
            overlap_ratio=overlap_ratio,
        )
        print(f"[{index}/{len(audio_files)}] Creato: {out_path}")


if __name__ == "__main__":
    inference_and_reconstruct_griffin_lim(
        input_dir=INPUT_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        output_audio_dir=OUTPUT_AUDIO_DIR,
        generator_key=GENERATOR_KEY,
        batch_size=BATCH_SIZE,
        griffin_lim_iters=GRIFFIN_LIM_ITERS,
        sr=RUN_SR,
        n_fft=RUN_N_FFT,
        hop_length=RUN_HOP_LENGTH,
        target_frames=RUN_TARGET_FRAMES,
        overlap_ratio=RUN_OVERLAP_RATIO,
    )
