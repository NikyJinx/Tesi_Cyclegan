import os
import glob
import numpy as np
import librosa


# Dataset builder stile Wav2NPY originale, ma con split in 2 file NPY per blocco:
# 1) ampiezza: shape (1, 512, T)
# 2) fase:     shape (2, 512, T) con [sin, cos]


def build_cyclegan_dataset_split(
    input_dir,
    output_amp_dir,
    output_phase_dir,
    sr=22050,
    n_fft=1024,
    hop_length=256,
    target_frames=2048,
    overlap_ratio=0.5,
    output_dtype=np.float16,
):
    """
    Legge i file WAV e salva due NPY per ogni blocco temporale:
    - *_amp.npy   con magnitudine normalizzata
    - *_phase.npy con sin e cos della fase

    Note:
    - n_fft=1024 produce 513 bin; qui salviamo 512 (rimuovendo Nyquist) come nel tuo schema.
    - target_frames controlla la larghezza (tempo) del blocco.
    """
    os.makedirs(output_amp_dir, exist_ok=True)
    os.makedirs(output_phase_dir, exist_ok=True)

    stride_frames = int(target_frames * (1.0 - overlap_ratio))
    if stride_frames <= 0:
        raise ValueError("overlap_ratio non valido: stride <= 0")

    duration_sec = (target_frames * hop_length) / sr
    print("\n=== CONFIGURAZIONE ===")
    print(f"Sample rate: {sr}")
    print(f"n_fft: {n_fft}")
    print(f"hop_length: {hop_length}")
    print(f"target_frames: {target_frames}")
    print(f"overlap_ratio: {overlap_ratio}")
    print(f"durata blocco: {duration_sec:.2f} s")
    print(f"dtype output: {np.dtype(output_dtype)}")
    print("======================\n")

    wav_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
    if not wav_files:
        print(f"Nessun file .wav trovato nella cartella: {input_dir}")
        return

    print(f"Trovati {len(wav_files)} file audio. Inizio conversione...")

    file_processati = 0
    blocchi_totali = 0

    for file_path in wav_files:
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

        # 4) Normalizzazione magnitudine come schema base
        # mag_db in [-80, 0], poi mag_norm in [-1, 1]
        mag_db = librosa.amplitude_to_db(magnitude, ref=np.max, top_db=80.0)
        mag_norm = (mag_db / 40.0) + 1.0

        sin_phase = np.sin(angle_rad)
        cos_phase = np.cos(angle_rad)

        # 5) Finestratura temporale
        total_frames = mag_norm.shape[1]
        if total_frames < target_frames:
            print(
                f"Saltato: {name} (troppo corto: {total_frames} frame, servono almeno {target_frames})"
            )
            continue

        starts = range(0, total_frames - target_frames + 1, stride_frames)
        num_blocks = len(range(0, total_frames - target_frames + 1, stride_frames))

        for i, start in enumerate(starts):
            end = start + target_frames

            amp_tensor = np.expand_dims(mag_norm[:, start:end], axis=0).astype(output_dtype)
            phase_tensor = np.stack(
                (sin_phase[:, start:end], cos_phase[:, start:end]), axis=0
            ).astype(output_dtype)

            amp_name = f"{name}_part_{i:03d}_amp.npy"
            phase_name = f"{name}_part_{i:03d}_phase.npy"

            np.save(os.path.join(output_amp_dir, amp_name), amp_tensor)
            np.save(os.path.join(output_phase_dir, phase_name), phase_tensor)
            blocchi_totali += 1

        file_processati += 1
        print(
            f"[{file_processati}/{len(wav_files)}] Elaborato: {name} -> Generati {num_blocks} blocchi"
        )

    print(
        f"\nOperazione completata! Creati {blocchi_totali} blocchi in coppie amp/phase."
    )


if __name__ == "__main__":
    CARTELLA_INPUT = "C:\\CycleGan\\STFT\\Audio_dominio_B_v2"

    # Cartelle output separate
    CARTELLA_OUTPUT_AMP = "E:\\Dataset_Tesi\\Dataset_B_V2_amp"
    CARTELLA_OUTPUT_PHASE = "E:\\Dataset_Tesi\\Dataset_B_V2_phase"

    # Parametri tempo/dimensione (qui cambi la durata)
    SR = 22050
    N_FFT = 1024
    HOP_LENGTH = 256
    TARGET_FRAMES = 2048      # 1024 o 2048 consigliati
    OVERLAP_RATIO = 0.5
    OUTPUT_DTYPE = np.float16

    build_cyclegan_dataset_split(
        CARTELLA_INPUT,
        CARTELLA_OUTPUT_AMP,
        CARTELLA_OUTPUT_PHASE,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        target_frames=TARGET_FRAMES,
        overlap_ratio=OVERLAP_RATIO,
        output_dtype=OUTPUT_DTYPE,
    )
