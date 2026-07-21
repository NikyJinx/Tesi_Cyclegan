import os
import glob
import numpy as np
import torch
import librosa
import soundfile as sf

# Importa il generatore usato in training (assicurati che la definizione del modello sia coerente con quella usata per il training e che il checkpoint contenga i pesi corretti).
from CycleGan_v4 import Generator_2_1_2D


# ==========================================
# WAV → NPY
# ==========================================

def wav_to_npy(input_dir, temp_dir):

    os.makedirs(temp_dir, exist_ok=True)

    sr = 22050
    n_fft = 1024
    hop_length = 256
    # hop_length = 512
    target_frames = 512

    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))

    created = 0

    for file_path in wav_files:

        name = os.path.splitext(os.path.basename(file_path))[0]

        y, _ = librosa.load(file_path, sr=sr)

        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

        D = D[:-1, :]

        mag = np.abs(D)
        phase = np.angle(D)

        mag_db = librosa.amplitude_to_db(mag, ref=np.max, top_db=80.0)

        mag_norm = (mag_db / 40.0) + 1.0

        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)

        total_frames = mag.shape[1]

        blocks = total_frames // target_frames

        for i in range(blocks):

            start = i * target_frames
            end = start + target_frames

            # Ordine canali coerente con il training: [Mag, Sin, Cos]
            tensor = np.stack([
                mag_norm[:, start:end],
                sin_phase[:, start:end],
                cos_phase[:, start:end]
            ], axis=0)

            save_name = f"{name}_part_{i:03d}.npy"

            np.save(os.path.join(temp_dir, save_name), tensor)

            created += 1

    return created


# ==========================================
# INFERENZA
# ==========================================

def inference_and_reconstruct(

        input_dir="input_audio",
        checkpoint_path="checkpoints/checkpoint_epoch_100.pt",
        output_audio_dir="output_audio",
    generator_key="G_AB",
    normalize_phase=True,
    use_original_phase=False

):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    os.makedirs(output_audio_dir, exist_ok=True)

    temp_dir = os.path.join(input_dir, "_temp_npy")
    os.makedirs(temp_dir, exist_ok=True)

    npy_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    created = wav_to_npy(input_dir, temp_dir)
    npy_from_wav = sorted(glob.glob(os.path.join(temp_dir, "*.npy")))

    npy_files = npy_files + npy_from_wav
    if not npy_files:
        raise RuntimeError("Nessun file .npy o .wav valido trovato nella cartella di input")

    if not npy_files:
        raise RuntimeError("Nessun spettrogramma trovato")

    # carica generatore

    G = Generator_2_1_2D().to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    G.load_state_dict(ckpt[generator_key])

    G.eval()

    print("Generatore caricato")
    if use_original_phase:
        print("Modalita fase: ORIGINALE input")
    elif normalize_phase:
        print("Modalita fase: GENERATA normalizzata")
    else:
        print("Modalita fase: GENERATA grezza")

    sr = 22050
    n_fft = 1024
    hop_length = 256
    # hop_length = 512 usare questa se dataset creato con questo hop_length

    with torch.no_grad():

        for file_path in npy_files:

            name = os.path.splitext(os.path.basename(file_path))[0]

            tensor = np.load(file_path).astype(np.float32)

            # Fase originale del blocco input (ordine canali: Mag, Sin, Cos)
            orig_sin_phase = tensor[1]
            orig_cos_phase = tensor[2]

            x = torch.from_numpy(tensor).unsqueeze(0).to(device)

            y = G(x)

            y = y.squeeze(0).cpu().numpy()

            mag_norm = y[0]
            sin_phase = y[1]
            cos_phase = y[2]

            # Sicurezza numerica: limita magnitudo in range atteso.
            mag_norm = np.clip(mag_norm, -1.0, 1.0)

            # Opzionale: usa fase originale input invece della fase generata.
            if use_original_phase:
                sin_phase = orig_sin_phase
                cos_phase = orig_cos_phase
            # Oppure rinormalizza la fase generata sul cerchio unitario.
            elif normalize_phase:
                phase_norm = np.sqrt((sin_phase ** 2) + (cos_phase ** 2) + 1e-8)
                sin_phase = sin_phase / phase_norm
                cos_phase = cos_phase / phase_norm

            mag_db = 40.0 * (mag_norm - 1.0)

            magnitude = librosa.db_to_amplitude(mag_db)

            phase = np.arctan2(sin_phase, cos_phase)

            D = magnitude * np.exp(1j * phase)

            D = np.pad(D, ((0, 1), (0, 0)))

            audio = librosa.istft(D, hop_length=hop_length, n_fft=n_fft)

            audio = librosa.util.normalize(audio)

            out_path = os.path.join(output_audio_dir, name + "_converted.wav")

            sf.write(out_path, audio, sr)

            print("Creato:", out_path)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    inference_and_reconstruct(
        input_dir="input_audio",
        checkpoint_path="checkpoints/checkpoint_epoch_100.pt",
        generator_key="G_AB",
        normalize_phase=True,
        use_original_phase=False

    )