import os
import glob
import math
import numpy as np
import librosa
import soundfile as sf

#Test per verificare la trasformaziuone da wav a tensore e viceversa, con i parametri fissi usati per la GAN.

# ==========================================
# MODULO 1: DA USARE SOLO PER CREARE IL DATASET
# ==========================================
def audio_to_tensors(input_dir, output_dir, sr=22050, n_fft=1024, hop_length=512, target_frames=512):
    """
    Legge gli audio e crea i tensori (3, 512, 512). 
    Se ne dimentica subito dopo averli salvati.
    """
    os.makedirs(output_dir, exist_ok=True)
    wav_files = glob.glob(os.path.join(input_dir, '*.wav'))
    
    for file_path in wav_files:
        name = os.path.splitext(os.path.basename(file_path))[0]
        y, _ = librosa.load(file_path, sr=sr)
        
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        D_512 = D[:-1, :] 
        
        magnitude = np.abs(D_512)
        angle_rad = np.angle(D_512)
        
        # Normalizzazione fissa e matematica (indipendente dal file)
        mag_db = librosa.amplitude_to_db(magnitude, ref=np.max, top_db=80.0) 
        mag_norm = (mag_db / 40.0) + 1.0 
        
        cos_phase = np.cos(angle_rad) 
        sin_phase = np.sin(angle_rad) 
        
        total_frames = mag_norm.shape[1]
        num_blocks = math.ceil(total_frames / target_frames)
        pad_width = (num_blocks * target_frames) - total_frames
        
        if pad_width > 0:
            mag_norm = np.pad(mag_norm, ((0, 0), (0, pad_width)), constant_values=-1.0)
            cos_phase = np.pad(cos_phase, ((0, 0), (0, pad_width)), constant_values=1.0)
            sin_phase = np.pad(sin_phase, ((0, 0), (0, pad_width)), constant_values=0.0)
            
        for i in range(num_blocks):
            start = i * target_frames
            end = start + target_frames
            tensor_3ch = np.stack((mag_norm[:, start:end], cos_phase[:, start:end], sin_phase[:, start:end]), axis=0)
            np.save(os.path.join(output_dir, f"{name}_part_{i:03d}.npy"), tensor_3ch)


# ==========================================
# MODULO 2: DA USARE DOPO LA CYCLEGAN (INFERENZA)
# ==========================================
def tensor_to_audio(tensor_path, output_wav_path, sr=22050, n_fft=1024, hop_length=512):
    """
    Funzione "cieca". Prende in input SOLO il tensore generato dalla GAN 
    e le costanti di architettura.
    """
    # 1. Carica solo i dati grezzi salvati (o generati dalla rete)
    tensor = np.load(tensor_path) # Shape: (3, 512, 512)
    mag_norm = tensor[0]
    cos_phase = tensor[1]
    sin_phase = tensor[2]
    
    # 2. Matematica inversa fissa (nessuna variabile dal modulo 1)
    mag_db = 40.0 * (mag_norm - 1.0)
    magnitude = librosa.db_to_amplitude(mag_db)
    
    # arctan2 ricava l'angolo in modo robusto anche se la GAN fa piccoli errori su cos/sin
    angle_rad = np.arctan2(sin_phase, cos_phase)
    
    # 3. Ricostruzione Complessa e iSTFT
    D_recon = magnitude * np.exp(1j * angle_rad)
    D_recon = np.pad(D_recon, ((0, 1), (0, 0)), mode='constant') # Aggiunge il bin di Nyquist
    y_recon = librosa.istft(D_recon, hop_length=hop_length, n_fft=n_fft)
    
    # 4. Normalizzazione autonoma: ascolta l'audio generato e lo alza al volume ottimale
    y_recon = librosa.util.normalize(y_recon)
    
    # 5. Salvataggio
    sf.write(output_wav_path, y_recon, sr)
    print(f"Audio generato in: {output_wav_path}")


# --- ESEMPIO DI UTILIZZO ---
if __name__ == "__main__":
    CARTELLA_INPUT = "input_audio"
    CARTELLA_TENSORS = "dataset_tensors"
    
    # 1. Creiamo il dataset dai file wav
    print("Avvio elaborazione dataset...")
    audio_to_tensors(CARTELLA_INPUT, CARTELLA_TENSORS)
    
    # 2. Troviamo automaticamente il primo file .npy appena creato
    file_generati = glob.glob(os.path.join(CARTELLA_TENSORS, '*.npy'))
    
    if file_generati:
        primo_file = file_generati[0]
        print(f"\nTest di inferenza sul file: {primo_file}")
        
        # 3. Testiamo la ricostruzione cieca
        tensor_to_audio(primo_file, "risultato_gan.wav")
    else:
        print(f"Nessun file .npy trovato. Assicurati di avere dei file .wav in {CARTELLA_INPUT}")