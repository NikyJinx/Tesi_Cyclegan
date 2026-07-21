import os
import glob
import numpy as np
import torch
import librosa
import soundfile as sf

# Importa il generatore usato in training v3
from CycleGan_v3 import Generator_2_1_2D


# ==========================================
# WAV → NPY
# ==========================================

def wav_to_npy(input_dir, temp_dir):

    os.makedirs(temp_dir, exist_ok=True)

    sr = 22050
    n_fft = 1024
    hop_length = 512
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

def apply_lofi_dsp(audio, sr, lowpass_hz=4500.0, saturation_drive=1.15, bit_depth=12):
    """Catena DSP lo-fi: low-pass + soft saturation + bitcrush leggero."""
    # 1) Low-pass in dominio frequenza per attenuare artefatti metallici sulle alte.
    cutoff = float(np.clip(lowpass_hz, 200.0, (sr * 0.5) - 100.0))
    spec = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(audio.shape[0], d=1.0 / sr)
    spec[freqs > cutoff] = 0.0
    filtered = np.fft.irfft(spec, n=audio.shape[0]).astype(np.float32)

    # 2) Saturazione morbida stile tape-like.
    drive = max(float(saturation_drive), 1.0)
    saturated = np.tanh(filtered * drive) / np.tanh(drive)

    # 3) Bitcrush leggero per colorazione lo-fi.
    bits = int(np.clip(bit_depth, 6, 16))
    levels = float((2 ** bits) - 1)
    crushed = np.round(saturated * levels) / levels
    return crushed.astype(np.float32)


def inference_and_reconstruct(

        input_dir="input_audio",
        checkpoint_path="checkpoints",
        output_audio_dir="output_audio_v2",
    generator_key="G_AB",
    normalize_phase=True,
    use_original_phase=False,
    use_lofi_dsp=True,
    lofi_lowpass_hz=3500.0,
    lofi_saturation_drive=1.3,
    lofi_bit_depth=10

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
    if use_lofi_dsp:
        print(
            f"DSP Lo-Fi: ON (low-pass={lofi_lowpass_hz:.0f} Hz, "
            f"drive={lofi_saturation_drive:.2f}, bit_depth={lofi_bit_depth})"
        )
    else:
        print("DSP Lo-Fi: OFF")

    sr = 22050
    n_fft = 1024
    hop_length = 512

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

            if use_lofi_dsp:
                audio = apply_lofi_dsp(
                    audio,
                    sr=sr,
                    lowpass_hz=lofi_lowpass_hz,
                    saturation_drive=lofi_saturation_drive,
                    bit_depth=lofi_bit_depth,
                )
                # Headroom per evitare clipping in export.
                peak = np.max(np.abs(audio)) + 1e-8
                audio = 0.95 * (audio / peak)

            out_path = os.path.join(output_audio_dir, name + "_converted.wav")

            sf.write(out_path, audio, sr)

            print("Creato:", out_path)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    inference_and_reconstruct(
        input_dir="input_audio",
        checkpoint_path="checkpoints/checkpoint_epoch_200.pt",
        generator_key="G_AB",
        normalize_phase=True,
        use_original_phase=False,
        use_lofi_dsp=True,
        lofi_lowpass_hz=3500.0,
        lofi_saturation_drive=1.1,
        lofi_bit_depth=12

    )
