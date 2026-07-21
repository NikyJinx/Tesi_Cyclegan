import os
import glob
import numpy as np
import librosa
from scipy import signal


#Creazione dataset per la CycleGAN: da wav a tensore (3, 512, 512) con Mag_norm migliorata.
#In particolare creando file .npy con blocchi 512x512 e overlap del 50% sull'asse temporale.
#MIGLIORAMENTI: normalizzazione logaritmica, smoothing temporale, compressione dinamica.

def build_cyclegan_dataset(input_dir, output_dir, use_magnitude_only=False, 
                           smooth_kernel=3, compress_dynamic_range=True, high_pass_cutoff=80):
    """
    Legge tutti i file .wav in input_dir, li trasforma in tensori 
    con rappresentazione migliorata e li salva come .npy in output_dir.
    
    Args:
        use_magnitude_only: Se True, usa solo 1 canale (magnitudo). Se False, 3 canali.
        smooth_kernel: Size del kernel per smoothing mediana (dispari). 0 = no smoothing.
        compress_dynamic_range: Applica compressione dinamica per migliorare contrasto.
        high_pass_cutoff: Freq di cutoff (Hz) per high-pass filter. 0 = nessun filtro.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Parametri architetturali
    sr = 22050
    n_fft = 1024
    hop_length = 256              
    target_frames = 2048
    stride_frames = target_frames // 2   # overlap 50%

    wav_files = glob.glob(os.path.join(input_dir, '*.wav'))
    if not wav_files:
        print(f"Nessun file .wav trovato nella cartella: {input_dir}")
        return

    print(f"Trovati {len(wav_files)} file audio. Inizio conversione...")
    print(f"Modalità: {'Magnitudo Only' if use_magnitude_only else '3-Canali'} | "
          f"Smoothing: {smooth_kernel}px | Compressione: {compress_dynamic_range}")
    
    file_processati = 0
    blocchi_totali = 0

    for file_path in wav_files:
        filename = os.path.basename(file_path)
        name, _ = os.path.splitext(filename)
        
        # 1. Caricamento audio
        y, _ = librosa.load(file_path, sr=sr)
        
        # 2. High-pass filter opzionale
        if high_pass_cutoff > 0:
            sos = signal.butter(4, high_pass_cutoff, 'hp', fs=sr, output='sos')
            y = signal.sosfilt(sos, y)
        
        # 3. STFT e taglio a 512 pixel di altezza
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        D_512 = D[:-1, :] 
        
        # 4. Separazione Magnitudo e Fase
        magnitude = np.abs(D_512)
        angle_rad = np.angle(D_512)
        
        # 5. Normalizzazione logaritmica migliorata
        mag_db = librosa.amplitude_to_db(magnitude, ref=np.max, top_db=100.0)  # Range più ampio
        # Clipping e normalizzazione a [-1, 1]
        mag_db = np.clip(mag_db, -100, 0)  # Clip ai dB
        mag_norm = (mag_db + 100) / 100.0 * 2.0 - 1.0  # Scale to [-1, 1]
        
        # 6. Smoothing temporale (mediana filter) per ridurre artefatti
        if smooth_kernel > 1:
            mag_norm = signal.medfilt2d(mag_norm, kernel_size=(1, smooth_kernel))
        
        # 7. Compressione dinamica (exponential compression per leggibilità)
        if compress_dynamic_range:
            mag_norm = np.sign(mag_norm) * np.power(np.abs(mag_norm), 0.8)
        
        # 8. Preparazione canali aggiuntivi
        cos_phase = np.cos(angle_rad) 
        sin_phase = np.sin(angle_rad)
        
        # 9. Applicare smoothing anche alla fase se in 3-canali
        if not use_magnitude_only and smooth_kernel > 1:
            sin_phase = signal.medfilt2d(sin_phase, kernel_size=(1, smooth_kernel))
            cos_phase = signal.medfilt2d(cos_phase, kernel_size=(1, smooth_kernel))
        
        # 10. Calcolo delle finestre con overlap del 50%
        total_frames = mag_norm.shape[1]
        if total_frames < target_frames:
            print(f"Saltato: {name} (troppo corto: {total_frames} frame, servono almeno {target_frames})")
            continue
        
        start_indices = range(0, total_frames - target_frames + 1, stride_frames)
        num_blocks = len(list(start_indices))
            
        # 11. Affettamento in blocchi e salvataggio
        for i, start in enumerate(start_indices):
            end = start + target_frames
            
            if use_magnitude_only:
                # Singolo canale: solo magnitudo
                tensor = mag_norm[:, start:end]
                tensor = np.expand_dims(tensor, axis=0)  # Aggiungi dimensione canale
            else:
                # 3 canali: magnitudo, sin_phase, cos_phase
                tensor = np.stack((
                    mag_norm[:, start:end], 
                    sin_phase[:, start:end],
                    cos_phase[:, start:end]
                ), axis=0)
            
            part_name = f"{name}_part_{i:03d}.npy"
            np.save(os.path.join(output_dir, part_name), tensor)
            blocchi_totali += 1
            
        file_processati += 1
        print(f"[{file_processati}/{len(wav_files)}] Elaborato: {name} -> Generati {num_blocks} blocchi.")

    print(f"\nOperazione completata! Creati in totale {blocchi_totali} file .npy in '{output_dir}'.")

# --- ESECUZIONE ---
if __name__ == "__main__":
    # Parametri configurabili
    CARTELLA_INPUT = "E:\\Dataset_Tesi\\Dataset_8Bit"
    CARTELLA_OUTPUT = "E:\\Dataset_Tesi\\dataset_dominio_B_8Bit"
    
    # Opzioni di elaborazione:
    USE_MAGNITUDE_ONLY = False      # Se True: solo magnitudo (1 canale). False: 3 canali (mag + sin/cos phase)
    SMOOTH_KERNEL = 5               # Size kernel mediana (dispari, es. 3,5,7). 0=no smoothing
    COMPRESS_DYNAMIC_RANGE = True   # Applica compressione per migliorare contrasto
    HIGH_PASS_CUTOFF = 80           # Hz, rimuove bassi (0=disabled)
    
    print("\n=== CONFIGURAZIONE ===")
    print(f"Magnitudo Only: {USE_MAGNITUDE_ONLY}")
    print(f"Smoothing kernel: {SMOOTH_KERNEL}px")
    print(f"Compressione dinamica: {COMPRESS_DYNAMIC_RANGE}")
    print(f"High-pass filter: {HIGH_PASS_CUTOFF} Hz")
    print("======================\n")
    
    build_cyclegan_dataset(
        CARTELLA_INPUT, 
        CARTELLA_OUTPUT,
        use_magnitude_only=USE_MAGNITUDE_ONLY,
        smooth_kernel=SMOOTH_KERNEL,
        compress_dynamic_range=COMPRESS_DYNAMIC_RANGE,
        high_pass_cutoff=HIGH_PASS_CUTOFF
    )