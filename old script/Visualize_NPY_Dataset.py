import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import warnings

warnings.filterwarnings('ignore')

# --- Percorsi dataset ---
domain_a_amp_dir = r"E:\Dataset_Tesi\Dataset_A_V2_amp"
domain_b_amp_dir = r"E:\Dataset_Tesi\Dataset_B_V2_amp"
domain_a_phase_dir = r"E:\Dataset_Tesi\Dataset_A_V2_phase"
domain_b_phase_dir = r"E:\Dataset_Tesi\Dataset_B_V2_phase"

def load_npy_files(directory, max_files=10):
    """Carica i primi N file NPY da una directory."""
    if not os.path.exists(directory):
        print(f"❌ Directory non trovata: {directory}")
        return []
    
    npy_files = sorted(glob.glob(os.path.join(directory, "*.npy")))[:max_files]
    print(f"✓ Trovati {len(npy_files)} file in {directory}")
    
    data = []
    for f in npy_files:
        try:
            arr = np.load(f)
            data.append((os.path.basename(f), arr))
        except Exception as e:
            print(f"  ⚠ Errore nel caricamento {os.path.basename(f)}: {e}")
    
    return data

print("=" * 80)
print("VISUALIZZAZIONE DATASET NPY")
print("=" * 80)

# Carica i dati
print("\n🔹 Caricamento ampiezza Dominio A...")
amp_a = load_npy_files(domain_a_amp_dir, 10)

print("\n🔹 Caricamento ampiezza Dominio B...")
amp_b = load_npy_files(domain_b_amp_dir, 10)

print("\n🔹 Caricamento fase Dominio A...")
phase_a = load_npy_files(domain_a_phase_dir, 10)

print("\n🔹 Caricamento fase Dominio B...")
phase_b = load_npy_files(domain_b_phase_dir, 10)

# --- Visualizzazione Ampiezza ---
if amp_a:
    print(f"\n🎨 Visualizzazione 10 immagini di AMPIEZZA...")
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Ampiezza - Dominio A (Dataset_V2_amp_A)", fontsize=14, fontweight='bold')
    axes = axes.flatten()
    
    for idx, (fname, arr) in enumerate(amp_a):
        ax = axes[idx]
        if arr.ndim == 3:
            arr = arr[0]  # Se ha 3 dimensioni, prendi il primo canale
        
        print(f"  [{idx+1}/10] {fname} - Shape: {arr.shape} - Min: {arr.min():.4f}, Max: {arr.max():.4f}")
        
        # Visualizza con scala logaritmica per miglior contrasto
        im = ax.imshow(arr, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(fname[:20], fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(r"C:\CycleGan\STFT\visualization_output\Amplitude_Dominio_A.png", dpi=100, bbox_inches='tight')
    print(f"\n✅ Salvato: visualization_output/Amplitude_Dominio_A.png")
    plt.show()

# --- Visualizzazione Fase (sin/cos) ---
if phase_a:
    print(f"\n🎨 Visualizzazione 10 immagini di FASE (Dominio A)...")
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    fig.suptitle("Fase - Dominio A (sin e cos) | Dataset_V2_phase_A", fontsize=14, fontweight='bold')
    axes = axes.flatten()
    
    for idx, (fname, arr) in enumerate(phase_a):
        print(f"  [{idx+1}/10] {fname} - Shape: {arr.shape}")
        
        # Canale 0: sin
        ax_sin = axes[idx * 2]
        if arr.ndim == 3:
            im = ax_sin.imshow(arr[0], aspect='auto', cmap='RdBu_r', origin='lower', vmin=-1, vmax=1)
            ax_sin.set_title(f"{fname[:15]} [SIN]", fontsize=8)
        ax_sin.axis('off')
        plt.colorbar(im, ax=ax_sin, fraction=0.046, pad=0.04)
        
        # Canale 1: cos
        ax_cos = axes[idx * 2 + 1]
        if arr.ndim == 3 and arr.shape[0] >= 2:
            im = ax_cos.imshow(arr[1], aspect='auto', cmap='RdBu_r', origin='lower', vmin=-1, vmax=1)
            ax_cos.set_title(f"{fname[:15]} [COS]", fontsize=8)
        ax_cos.axis('off')
        plt.colorbar(im, ax=ax_cos, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(r"C:\CycleGan\STFT\visualization_output\Phase_Dominio_A_SinCos.png", dpi=100, bbox_inches='tight')
    print(f"\n✅ Salvato: visualization_output/Phase_Dominio_A_SinCos.png")
    plt.show()

# --- Statistiche ---
print("\n" + "=" * 80)
print("📊 STATISTICHE DATASET")
print("=" * 80)

if amp_a:
    amp_array = amp_a[0][1]
    print(f"\n🔷 Ampiezza Dominio A:")
    print(f"   Shape: {amp_array.shape}")
    print(f"   Min/Max: [{amp_array.min():.6f}, {amp_array.max():.6f}]")
    print(f"   Mean/Std: {amp_array.mean():.6f} ± {amp_array.std():.6f}")

if phase_a:
    phase_array = phase_a[0][1]
    print(f"\n🔷 Fase Dominio A:")
    print(f"   Shape: {phase_array.shape}")
    print(f"   Min/Max: [{phase_array.min():.6f}, {phase_array.max():.6f}]")
    print(f"   Mean/Std: {phase_array.mean():.6f} ± {phase_array.std():.6f}")

print("\n" + "=" * 80)
