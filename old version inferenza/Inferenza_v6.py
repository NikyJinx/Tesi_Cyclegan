import argparse
import os
from typing import List, Sequence, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch


# Scegli il modello

from CycleGan_v6 import Generator_2_1_2D


SR = 22050
N_FFT = 1024
HOP_LENGTH = 256
TARGET_FRAMES = 2048
OVERLAP_RATIO = 0.5
TOP_DB = 80.0
PEAK_HEADROOM = 0.95
SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".opus", ".flac")
DEFAULT_USE_ORIGINAL_PHASE = False


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
) -> Tuple[np.ndarray, List[int], int, np.ndarray, np.ndarray]:
    stride_frames = int(target_frames * (1.0 - overlap_ratio))
    if stride_frames <= 0:
        raise ValueError("overlap_ratio non valido: stride <= 0")

    audio, _ = librosa.load(file_path, sr=sr)
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

    return np.stack(blocks, axis=0), starts, len(audio), sin_phase, cos_phase


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


def merge_phase_blocks(
    predicted_blocks: np.ndarray,
    starts: Sequence[int],
    total_frames: int,
    target_frames: int,
    normalize_phase: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    merged_sin = np.zeros((512, total_frames), dtype=np.float32)
    merged_cos = np.zeros((512, total_frames), dtype=np.float32)
    weights = np.zeros(total_frames, dtype=np.float32)

    for block_idx, start in enumerate(starts):
        end = min(start + target_frames, total_frames)
        valid_width = end - start

        merged_sin[:, start:end] += predicted_blocks[block_idx, 1, :, :valid_width]
        merged_cos[:, start:end] += predicted_blocks[block_idx, 2, :, :valid_width]
        weights[start:end] += 1.0

    weights = np.maximum(weights, 1e-8)
    merged_sin = merged_sin / weights[np.newaxis, :]
    merged_cos = merged_cos / weights[np.newaxis, :]

    if normalize_phase:
        phase_norm = np.sqrt((merged_sin ** 2) + (merged_cos ** 2) + 1e-8)
        merged_sin = merged_sin / phase_norm
        merged_cos = merged_cos / phase_norm

    return merged_sin, merged_cos


def reconstruct_audio(
    merged_mag_norm: np.ndarray,
    sin_phase: np.ndarray,
    cos_phase: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    target_length: int,
) -> np.ndarray:
    merged_mag_norm = np.clip(merged_mag_norm, -1.0, 1.0)
    mag_db = 40.0 * (merged_mag_norm - 1.0)
    magnitude = librosa.db_to_amplitude(mag_db)

    phase = np.arctan2(sin_phase, cos_phase)
    complex_stft = magnitude * np.exp(1j * phase)
    complex_stft = np.pad(complex_stft, ((0, 1), (0, 0)))

    audio = librosa.istft(complex_stft, hop_length=hop_length, n_fft=n_fft, length=target_length)

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
    use_original_phase: bool,
    normalize_phase: bool,
    sr: int,
    n_fft: int,
    hop_length: int,
    target_frames: int,
    overlap_ratio: float,
) -> str:
    blocks, starts, audio_length, orig_sin, orig_cos = build_input_blocks(
        file_path=file_path,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        target_frames=target_frames,
        overlap_ratio=overlap_ratio,
    )

    predictions = run_generator(generator, blocks, device=device, batch_size=batch_size)
    total_frames = orig_sin.shape[1]
    merged_mag_norm = merge_magnitude_blocks(predictions, starts, total_frames, target_frames)

    if use_original_phase:
        sin_phase = orig_sin
        cos_phase = orig_cos
    else:
        sin_phase, cos_phase = merge_phase_blocks(
            predictions,
            starts,
            total_frames,
            target_frames,
            normalize_phase=normalize_phase,
        )

    audio = reconstruct_audio(
        merged_mag_norm=merged_mag_norm,
        sin_phase=sin_phase,
        cos_phase=cos_phase,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        target_length=audio_length,
    )

    input_name = os.path.splitext(os.path.basename(file_path))[0]
    out_name = f"{input_name}_{generator_key}_converted.wav"
    out_path = os.path.join(output_dir, out_name)
    sf.write(out_path, audio, sr)
    return out_path


def inference_and_reconstruct(
    input_dir: str = "C:\\CycleGan\\STFT\\Input",
    checkpoint_path: str = "C:\\CycleGan\\STFT\\checkpoints\\checkpoint_epoch_093.pt",
    output_audio_dir: str = "C:\\CycleGan\\STFT\\Output",
    generator_key: str = "G_AB",
    batch_size: int = 1,
    normalize_phase: bool = True,
    use_original_phase: bool = False,
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
    if use_original_phase:
        print("Modalita fase: ORIGINALE input")
    elif normalize_phase:
        print("Modalita fase: GENERATA normalizzata")
    else:
        print("Modalita fase: GENERATA grezza")

    for index, file_path in enumerate(audio_files, start=1):
        out_path = infer_file(
            file_path=file_path,
            generator=generator,
            output_dir=output_audio_dir,
            device=device,
            generator_key=generator_key,
            batch_size=batch_size,
            use_original_phase=use_original_phase,
            normalize_phase=normalize_phase,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            target_frames=target_frames,
            overlap_ratio=overlap_ratio,
        )
        print(f"[{index}/{len(audio_files)}] Creato: {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inferenza audio per CycleGan_v6")
    parser.add_argument("--input-dir", default="C:\\CycleGan\\STFT\\Input", help="Cartella con file audio in input")
    parser.add_argument(
        "--checkpoint-path",
        default="C:\\CycleGan\\STFT\\checkpoints\\checkpoint_epoch_093.pt",
        help="Checkpoint contenente G_AB o G_BA",
    )
    parser.add_argument(
        "--output-audio-dir",
        default="C:\\CycleGan\\STFT\\Output",
        help="Cartella dove salvare i WAV convertiti",
    )
    parser.add_argument(
        "--generator-key",
        default="G_AB",
        choices=("G_AB", "G_BA"),
        help="Generatore da usare dal checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per i blocchi di inferenza")
    parser.add_argument(
        "--use-original-phase",
        action="store_true",
        help="Forza l'uso della fase originale dell'input",
    )
    parser.add_argument(
        "--use-generated-phase",
        action="store_true",
        help="Forza l'uso della fase generata dal modello",
    )
    parser.add_argument(
        "--skip-phase-normalization",
        action="store_true",
        help="Non rinormalizzare sin/cos quando si usa la fase generata",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    if args.use_original_phase and args.use_generated_phase:
        raise ValueError("Non puoi usare insieme --use-original-phase e --use-generated-phase")

    if args.use_original_phase:
        use_original_phase = True
    elif args.use_generated_phase:
        use_original_phase = False
    else:
        # Default usato quando avvii con "Run Python File" senza argomenti CLI.
        use_original_phase = DEFAULT_USE_ORIGINAL_PHASE

    inference_and_reconstruct(
        input_dir=args.input_dir,
        checkpoint_path=args.checkpoint_path,
        output_audio_dir=args.output_audio_dir,
        generator_key=args.generator_key,
        batch_size=args.batch_size,
        normalize_phase=not args.skip_phase_normalization,
        use_original_phase=use_original_phase,
    )