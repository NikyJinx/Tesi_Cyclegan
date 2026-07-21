import argparse
import glob
import os
from typing import List, Tuple

import librosa
import numpy as np
import tifffile


DEFAULT_INPUT_DIR = r"C:\CycleGan\STFT\TEST"
DEFAULT_TIFF_OUTPUT_DIR = r"C:\CycleGan\STFT\TEST\PNG"
DEFAULT_OUTPUT_FORMAT = "png"


def _list_audio_files(input_dir: str, supported_extensions: Tuple[str, ...]) -> List[str]:
    supported = tuple(ext.lower() for ext in supported_extensions)
    return sorted(
        entry.path
        for entry in os.scandir(input_dir)
        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in supported
    )


def _parse_selection(selection: str, files: List[str]) -> List[str]:
    if not selection.strip():
        return files

    chosen: List[str] = []
    tokens = [token.strip() for token in selection.split(",") if token.strip()]
    file_map = {os.path.basename(path): path for path in files}

    for token in tokens:
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(files):
                chosen.append(files[idx])
            else:
                print(f"Indice fuori range ignorato: {token}")
        elif token in file_map:
            chosen.append(file_map[token])
        else:
            print(f"File non trovato nella cartella (ignorato): {token}")

    deduped: List[str] = []
    seen = set()
    for path in chosen:
        if path not in seen:
            seen.add(path)
            deduped.append(path)

    return deduped


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


def build_cyclegan_tensor_blocks(
    file_path: str,
    sr: int,
    n_fft: int,
    hop_length: int,
    target_frames: int,
    overlap_ratio: float,
) -> Tuple[np.ndarray, List[int], int]:
    stride_frames = int(target_frames * (1.0 - overlap_ratio))
    if stride_frames <= 0:
        raise ValueError("overlap_ratio non valido: stride <= 0")

    audio, _ = librosa.load(file_path, sr=sr)
    stft_matrix = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    stft_matrix = stft_matrix[:-1, :]

    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    mag_db = librosa.amplitude_to_db(magnitude, ref=np.max, top_db=80.0)
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

        tensor_3ch = np.stack((mag_block, sin_block, cos_block), axis=0).astype(np.float32)
        blocks.append(tensor_3ch)

    return np.stack(blocks, axis=0), starts, len(audio)


def build_amplitude_matrix(
    audio: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    stft_matrix = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    stft_matrix = stft_matrix[:-1, :]
    magnitude = np.abs(stft_matrix)
    amp_db = librosa.amplitude_to_db(magnitude, ref=np.max, top_db=80.0)

    return np.clip((amp_db / 40.0) + 1.0, -1.0, 1.0).astype(np.float32)


def build_mel_amplitude_matrix(
    audio: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int = 128,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    amp_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    return np.clip((amp_db / 40.0) + 1.0, -1.0, 1.0).astype(np.float32)


def _tensor_to_images(tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tensor.ndim != 3:
        raise ValueError(f"Shape non supportata: {tensor.shape}. Atteso tensore 3D (3, H, W).")
    if tensor.shape[0] != 3:
        raise ValueError(f"Primo asse non valido: {tensor.shape}. Atteso primo asse = 3 canali.")

    magnitude = tensor[0].astype(np.float32)
    sin_phase = tensor[1].astype(np.float32)
    cos_phase = tensor[2].astype(np.float32)
    phase = np.arctan2(sin_phase, cos_phase).astype(np.float32)
    image_3layer = np.transpose(tensor.astype(np.float32), (1, 2, 0))

    return magnitude, phase, image_3layer


def _to_uint8(arr: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    clipped = np.clip(arr, min_value, max_value)
    normalized = (clipped - min_value) / (max_value - min_value + 1e-12)
    return (normalized * 255.0).astype(np.uint8)


def _to_uint16(arr: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    clipped = np.clip(arr, min_value, max_value)
    normalized = (clipped - min_value) / (max_value - min_value + 1e-12)
    return (normalized * 65535.0).astype(np.uint16)


def save_image(
    path_without_ext: str,
    image_data: np.ndarray,
    output_format: str,
    data_min: float,
    data_max: float,
    jpeg_quality: int,
    webp_quality: int,
) -> str:
    output_format = output_format.lower()
    ext = "jpg" if output_format == "jpeg" else output_format
    out_path = f"{path_without_ext}.{ext}"

    if output_format == "tiff":
        tifffile.imwrite(out_path, image_data.astype(np.float32))
        return out_path

    if output_format == "png":
        # PNG 16-bit per mantenere piu dinamica rispetto a 8-bit.
        tifffile.imwrite(out_path, _to_uint16(image_data, data_min, data_max))
        return out_path

    if image_data.ndim == 2:
        image_u8 = _to_uint8(image_data, data_min, data_max)
        from PIL import Image
        pil_image = Image.fromarray(image_u8, mode="L")
    elif image_data.ndim == 3 and image_data.shape[2] == 3:
        image_u8 = _to_uint8(image_data, data_min, data_max)
        from PIL import Image
        pil_image = Image.fromarray(image_u8, mode="RGB")
    else:
        raise ValueError(f"Shape immagine non supportata per export: {image_data.shape}")

    save_kwargs = {}
    if output_format in ("jpg", "jpeg"):
        save_kwargs["quality"] = int(np.clip(jpeg_quality, 1, 100))
        save_kwargs["optimize"] = True
        pil_image.save(out_path, format="JPEG", **save_kwargs)
    elif output_format == "webp":
        save_kwargs["quality"] = int(np.clip(webp_quality, 1, 100))
        save_kwargs["method"] = 6
        pil_image.save(out_path, format="WEBP", **save_kwargs)
    else:
        raise ValueError(f"Formato non supportato: {output_format}")

    return out_path


def export_wav_to_npy_and_tiff(
    input_dir: str,
    tiff_output_dir: str,
    selected: List[str] | None = None,
    sr: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    target_frames: int = 2048,
    overlap_ratio: float = 0.5,
    use_mel_amplitude: bool = True,
    mel_bins: int = 128,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    jpeg_quality: int = 90,
    webp_quality: int = 85,
    supported_extensions: Tuple[str, ...] = (".wav", ".mp3", ".opus", ".flac"),
) -> int:
    all_files = _list_audio_files(input_dir, supported_extensions)
    if not all_files:
        raise FileNotFoundError(f"Nessun file audio trovato in: {input_dir}")

    files = selected if selected is not None else all_files
    os.makedirs(tiff_output_dir, exist_ok=True)
    amplitude_dir = os.path.join(tiff_output_dir, "amplitude")
    phase_dir = os.path.join(tiff_output_dir, "phase")
    layer_dir = os.path.join(tiff_output_dir, "3layer")
    os.makedirs(amplitude_dir, exist_ok=True)
    os.makedirs(phase_dir, exist_ok=True)
    os.makedirs(layer_dir, exist_ok=True)
    mel_amplitude_dir = os.path.join(tiff_output_dir, "amplitude_mel128")
    if use_mel_amplitude:
        os.makedirs(mel_amplitude_dir, exist_ok=True)

    converted = 0
    for file_path in files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        source_amplitude_dir = os.path.join(amplitude_dir, base_name)
        source_phase_dir = os.path.join(phase_dir, base_name)
        source_layer_dir = os.path.join(layer_dir, base_name)
        os.makedirs(source_amplitude_dir, exist_ok=True)
        os.makedirs(source_phase_dir, exist_ok=True)
        os.makedirs(source_layer_dir, exist_ok=True)

        source_mel_dir = None
        if use_mel_amplitude:
            source_mel_dir = os.path.join(mel_amplitude_dir, base_name)
            os.makedirs(source_mel_dir, exist_ok=True)

        try:
            audio, _ = librosa.load(file_path, sr=sr)
            tensors, starts, _ = build_cyclegan_tensor_blocks(
                file_path=file_path,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                target_frames=target_frames,
                overlap_ratio=overlap_ratio,
            )
            amplitude_matrix = build_amplitude_matrix(
                audio=audio,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            mel_amplitude_matrix = None
            if use_mel_amplitude:
                mel_amplitude_matrix = build_mel_amplitude_matrix(
                    audio=audio,
                    sr=sr,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=mel_bins,
                )
        except Exception as exc:
            print(f"Errore su {os.path.basename(file_path)}: {exc}")
            continue

        exported_blocks = 0
        for block_idx, tensor in enumerate(tensors):
            try:
                magnitude, phase, image_3layer = _tensor_to_images(tensor)
            except Exception as exc:
                print(f"Errore nella conversione TIFF di {base_name}_part_{block_idx:03d}: {exc}")
                continue

            start = starts[block_idx]
            end = min(start + target_frames, amplitude_matrix.shape[1])
            amplitude_block = pad_block(
                amplitude_matrix[:, start:end],
                -1.0,
                target_frames,
            )

            tiff_base = f"{base_name}_part_{block_idx:03d}"
            amp_base = os.path.join(source_amplitude_dir, f"{tiff_base}_amplitude")
            phase_base = os.path.join(source_phase_dir, f"{tiff_base}_phase")
            layer_base = os.path.join(source_layer_dir, f"{tiff_base}_3layer")

            amp_path = save_image(
                path_without_ext=amp_base,
                image_data=amplitude_block,
                output_format=output_format,
                data_min=-1.0,
                data_max=1.0,
                jpeg_quality=jpeg_quality,
                webp_quality=webp_quality,
            )
            if mel_amplitude_matrix is not None:
                mel_end = min(start + target_frames, mel_amplitude_matrix.shape[1])
                mel_block = pad_block(
                    mel_amplitude_matrix[:, start:mel_end],
                    -1.0,
                    target_frames,
                )
                mel_base = os.path.join(source_mel_dir, f"{tiff_base}_mel128")
                mel_amp_path = save_image(
                    path_without_ext=mel_base,
                    image_data=mel_block,
                    output_format=output_format,
                    data_min=-1.0,
                    data_max=1.0,
                    jpeg_quality=jpeg_quality,
                    webp_quality=webp_quality,
                )
            phase_path = save_image(
                path_without_ext=phase_base,
                image_data=phase,
                output_format=output_format,
                data_min=-np.pi,
                data_max=np.pi,
                jpeg_quality=jpeg_quality,
                webp_quality=webp_quality,
            )
            layer_path = save_image(
                path_without_ext=layer_base,
                image_data=image_3layer,
                output_format=output_format,
                data_min=-1.0,
                data_max=1.0,
                jpeg_quality=jpeg_quality,
                webp_quality=webp_quality,
            )

            exported_blocks += 1
            converted += 1
            print(
                f"Creati: {os.path.basename(amp_path)}, {os.path.basename(phase_path)}, "
                f"{os.path.basename(layer_path)}"
            )
            if mel_amplitude_matrix is not None:
                print(f"        + {os.path.basename(mel_amp_path)}")

        print(f"File completato: {base_name} -> {exported_blocks} blocchi")

    return converted


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Legge WAV e li esporta in immagini (PNG/JPG/WEBP/TIFF) per ampiezza, fase e 3 layer"
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Cartella con i file audio in input",
    )
    parser.add_argument(
        "--tiff-output-dir",
        default=DEFAULT_TIFF_OUTPUT_DIR,
        help="Cartella dove salvare i TIFF generati",
    )
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate di lettura")
    parser.add_argument("--n-fft", type=int, default=1024, help="Dimensione FFT")
    parser.add_argument("--hop-length", type=int, default=256, help="Hop length STFT")
    parser.add_argument("--target-frames", type=int, default=2048, help="Numero di frame per blocco")
    parser.add_argument("--overlap-ratio", type=float, default=0.5, help="Rapporto di sovrapposizione tra blocchi")
    parser.add_argument(
        "--output-format",
        type=str,
        default=DEFAULT_OUTPUT_FORMAT,
        choices=("png", "jpg", "webp", "tiff"),
        help="Formato immagine di output (piu leggero: png/jpg/webp)",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90, help="Qualita JPG (1-100)")
    parser.add_argument("--webp-quality", type=int, default=85, help="Qualita WEBP (1-100)")
    parser.add_argument(
        "--use-mel-amplitude",
        action="store_true",
        help="Crea le immagini di amplitude usando una mel-spectrogram a 128 bande",
    )
    parser.set_defaults(use_mel_amplitude=True)
    parser.add_argument(
        "--selection",
        default="",
        help="Selezione file: vuoto = tutti, oppure 1,3,8 / nomefile.wav / misto",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Cartella input inesistente: {args.input_dir}")
        return

    audio_files = _list_audio_files(args.input_dir, (".wav", ".mp3", ".opus", ".flac"))
    if not audio_files:
        print("Nessun file audio trovato.")
        return

    print("=== WAV -> IMAGES CycleGAN v6 ===")
    print(f"Input: {args.input_dir}")
    print(f"Output TIFF: {args.tiff_output_dir}")
    print(f"Formato output: {args.output_format}")
    print("Sottocartelle TIFF:")
    print("- amplitude/")
    print("- phase/")
    print("- 3layer/")
    if args.use_mel_amplitude:
        print("- amplitude_mel128/ (attiva)")
    else:
        print("- amplitude_mel128/ (disattivata)")
    print(f"File trovati: {len(audio_files)}")
    print("Output per ogni blocco:")
    print(f"- *_amplitude.{args.output_format}")
    print(f"- *_phase.{args.output_format}")
    print(f"- *_3layer.{args.output_format}")

    chosen_files = _parse_selection(args.selection, audio_files)
    if not chosen_files:
        print("Nessun file valido selezionato.")
        return

    print(f"File selezionati: {len(chosen_files)}")
    converted = export_wav_to_npy_and_tiff(
        input_dir=args.input_dir,
        tiff_output_dir=args.tiff_output_dir,
        selected=chosen_files,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        target_frames=args.target_frames,
        overlap_ratio=args.overlap_ratio,
        use_mel_amplitude=args.use_mel_amplitude,
        output_format=args.output_format,
        jpeg_quality=args.jpeg_quality,
        webp_quality=args.webp_quality,
    )
    print(f"\nCompletato. Convertiti {converted} blocchi in: {args.tiff_output_dir}")


if __name__ == "__main__":
    main()
