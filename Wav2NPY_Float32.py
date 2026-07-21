import os
import glob
import numpy as np
import librosa


# Versione migliorata di Wav2NPY con float32 e normalizzazione ottimizzata
# - float32 per evitare quantizzazione
# - Normalizzazione Tanh per miglior range dinamico
# - Opzione di clipping soft per preservare dinamica


def build_cyclegan_dataset_float32(
    input_dir,
    output_dir,
    sr=22050,
    n_fft=1024,
    hop_length=256,
    target_frames=2048,
    overlap_ratio=0.5,
    supported_extensions=(".wav", ".mp3", ".opus"),
):
    """
    Versione compatibile con CycleGan_v6:
    - Output dtype: float32
    - Un solo file .npy per blocco con shape (3, 512, 2048): [mag, sin, cos]
    - Magnitudine normalizzata in [-1, 1] con silenzio circa -1 e picco circa +1
    """
    os.makedirs(output_dir, exist_ok=True)

    stride_frames = int(target_frames * (1.0 - overlap_ratio))
    if stride_frames <= 0:
        raise ValueError("overlap_ratio non valido: stride <= 0")

    duration_sec = (target_frames * hop_length) / sr
    print("\n=== CONFIGURAZIONE DATASET MIGLIORATO ===")
    print(f"Sample rate: {sr}")
    print(f"n_fft: {n_fft}")
    print(f"hop_length: {hop_length}")
    print(f"target_frames: {target_frames}")
    print(f"overlap_ratio: {overlap_ratio}")
    print(f"durata blocco: {duration_sec:.2f} s")
    print(f"dtype output: float32")
    print(f"shape output: (3, 512, {target_frames})")
    print(f"normalizzazione magnitudine: lineare compatibile con CycleGan_v6")
    print("=" * 45 + "\n")

    supported_extensions = tuple(ext.lower() for ext in supported_extensions)
    audio_files = sorted(
        entry.path
        for entry in os.scandir(input_dir)
        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in supported_extensions
    )

    if not audio_files:
        print(
            f"Nessun file audio trovato in: {input_dir} "
            f"(estensioni supportate: {', '.join(supported_extensions)})"
        )
        return

    print(f"Trovati {len(audio_files)} file audio. Inizio conversione...")

    file_processati = 0
    blocchi_totali = 0

    for file_path in audio_files:
        filename = os.path.basename(file_path)
        name, _ = os.path.splitext(filename)

        # 1) Caricamento audio
        y, _ = librosa.load(file_path, sr=sr)

        # 2) STFT e rimozione riga Nyquist (513 -> 512)
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        D_512 = D[:-1, :]

        # 3) Magnitudine e fase
        magnitude = np.abs(D_512)
        angle_rad = np.angle(D_512)

        # 4) Normalizzazione magnitudine compatibile con CycleGan_v6
        # mag_db e in [-80, 0] (top_db=80), quindi:
        # -80 dB -> -1 (silenzio), 0 dB -> +1 (picco locale)
        mag_db = librosa.amplitude_to_db(magnitude, ref=np.max, top_db=80.0)
        mag_norm = np.clip((mag_db / 40.0) + 1.0, -1.0, 1.0)
        
        sin_phase = np.sin(angle_rad)
        cos_phase = np.cos(angle_rad)

        # 5) Finestratura temporale
        total_frames = mag_norm.shape[1]
        if total_frames < target_frames:
            print(
                f"  ⚠ Saltato: {name} (troppo corto: {total_frames} frame, servono almeno {target_frames})"
            )
            continue

        starts = range(0, total_frames - target_frames + 1, stride_frames)
        num_blocks = len(range(0, total_frames - target_frames + 1, stride_frames))

        for i, start in enumerate(starts):
            end = start + target_frames

            # Salva un unico tensore a 3 canali: [mag, sin, cos]
            tensor_3ch = np.stack(
                (
                    mag_norm[:, start:end],
                    sin_phase[:, start:end],
                    cos_phase[:, start:end],
                ),
                axis=0,
            ).astype(np.float32)

            sample_name = f"{name}_part_{i:03d}.npy"

            np.save(os.path.join(output_dir, sample_name), tensor_3ch)
            blocchi_totali += 1

        file_processati += 1
        print(
            f"  ✓ [{file_processati}/{len(audio_files)}] {name} -> {num_blocks} blocchi"
        )

    print(
        f"\n✅ Completato! Creati {blocchi_totali} blocchi (float32)."
    )


if __name__ == "__main__":
    # Modifica qui il dominio (A or B)
    DOMINIO = "B"  # o "B"
    
    INPUT_DIR = f"C:\\CycleGan\\STFT\\Audio_Dataset_Violin_Jazz"
    OUTPUT_DIR = f"E:\\Dataset_Tesi\\dataset_dominio_Violin_Jazz"

    SR = 22050
    N_FFT = 1024
    HOP_LENGTH = 256
    TARGET_FRAMES = 2048
    OVERLAP_RATIO = 0.5

    build_cyclegan_dataset_float32(
        INPUT_DIR,
        OUTPUT_DIR,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        target_frames=TARGET_FRAMES,
        overlap_ratio=OVERLAP_RATIO,
    )
