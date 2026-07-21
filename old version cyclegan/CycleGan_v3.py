import os
import glob
import time
import random
import sys
import contextlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Generatore 2-1-2D CNN con GLU
# Questo generatore accetta il tensore a 3 canali, lo comprime spazialmente (asse frequenze) 
# tramite CNN 2D, opera la trasformazione temporale tramite CNN 1D con Gated Linear Units e 
# infine lo riespande.

class ResidualPseudo1DBlock_GLU(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        # Conv2d con kernel (1, kernel_size): opera SOLO sull'asse del tempo,
        # lasciando le frequenze indipendenti. Stessa logica di una Conv1d
        # ma senza fondere canali e frequenze (risparmio enorme di parametri).
        self.conv = nn.Conv2d(channels, channels * 2, kernel_size=(1, kernel_size), padding=(0, kernel_size//2))
        self.norm = nn.InstanceNorm2d(channels * 2)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.norm(out)
        
        # Gated Linear Unit
        out, gate = out.chunk(2, dim=1)
        out = out * torch.sigmoid(gate)
        return residual + out

class Generator_2_1_2D(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Fase 1: Downsampling 2D (es. da 512x512 a 128x128)
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True)
        ) # Output:
        
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ) # Output:

        # Fase 2: Collo di bottiglia Pseudo-1D (sull'asse del tempo)
        # Il tensore resta 4D (B, 64, Freq, Time). Il kernel (1,3) elabora solo il tempo.
        self.res_blocks = nn.Sequential(
            *[ResidualPseudo1DBlock_GLU(64) for _ in range(6)]
        )

        # Fase 3: Upsampling 2D (ricostruzione a 512x512)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True)
        ) # Output:
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(32, in_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        ) # Output:

    def forward(self, x):
        # Elaborazione Spaziale 2D
        x = self.down1(x)
        x = self.down2(x)

        # Elaborazione Temporale Pseudo-1D (il tensore resta 4D)
        x = self.res_blocks(x)

        # Ricostruzione Spaziale 2D
        x = self.up1(x)
        x = self.up2(x)
        
        # Output: tutti e 3 i canali in [-1, 1] con Tanh
        # (la magnitudine normalizzata vive in [-1, 1], NON in [0, ∞), quindi ReLU sarebbe sbagliato)
        return torch.tanh(x)


# Multi-Scale Discriminator (MSD)
# Per valutare coerenze sia macroscopiche (il ritmo globale) che microscopiche 
# (la fase e le alte frequenze), instanziamo più discriminatori su scale diverse 
# tramite subsampling.

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc_high = PatchGANDiscriminator()  # Scala originale
        self.disc_mid = PatchGANDiscriminator()   # Scala 1/2
        self.disc_low = PatchGANDiscriminator()   # Scala 1/4
        
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out_high = self.disc_high(x)
        
        x_mid = self.downsample(x)
        out_mid = self.disc_mid(x_mid)
        
        x_low = self.downsample(x_mid)
        out_low = self.disc_low(x_low)
        
        return [out_high, out_mid, out_low]


# Funzioni di Perdita Avanzate (Losses)
# Loss personalizzate che costringono il modello a rispettare l'identità trigonometrica
# sin^2(theta) + cos^2(theta) = 1 e applicano il vincolo solo dove c'è reale energia
# acustica (mascherando il rumore generato nel silenzio).

def trigonometric_consistency_loss(pred_sin, pred_cos):
    """
    Forza Seno e Coseno generati a giacere sul cerchio unitario.
    """
    squared_sum = torch.pow(pred_sin, 2) + torch.pow(pred_cos, 2)
    loss = torch.mean(torch.abs(squared_sum - 1.0))
    return loss

def magnitude_aware_cycle_loss(pred_x, real_x):
    """
    Cycle Consistency Loss pesata sulla magnitudo reale per mascherare i gradienti
    caotici del rumore di fondo.
    pred_x: output generato dal ciclo F(G(x))
    real_x: input originario x
    """
    # Indici canali: 0 -> Mag, 1 -> Sin, 2 -> Cos
    pred_mag, pred_sin, pred_cos = pred_x[:, 0:1], pred_x[:, 1:2], pred_x[:, 2:3]
    real_mag, real_sin, real_cos = real_x[:, 0:1], real_x[:, 1:2], real_x[:, 2:3]
    
    # Loss standard L1 sulla magnitudo
    loss_mag = torch.mean(torch.abs(pred_mag - real_mag))
    
    # Loss di fase moltiplicata per la magnitudo dell'input REALE
    # In questo modo si ignora l'errore di fase nelle zone in cui real_mag è ~0
    error_sin = torch.abs(pred_sin - real_sin)
    error_cos = torch.abs(pred_cos - real_cos)
    
    # Il distacco (.detach()) sulla magnitudo evita che il calcolo dei gradienti di fase 
    # alteri forzatamente i pesi deputati a generare la magnitudo.
    # Rimappiamo [-1, 1] -> [0, 1]: silenzio (-1) diventa peso 0, picco (+1) diventa peso 1
    weight_mask = (real_mag.detach() + 1.0) / 2.0
    loss_phase = torch.mean((error_sin + error_cos) * weight_mask)
    
    return loss_mag + loss_phase

def lsgan_loss(disc_preds, is_real):
    """
    Least Squares GAN Loss applicata a tutti gli output del discriminatore multi-scala.
    """
    loss = 0.0
    target_val = 0.9 if is_real else 0.1  # anche per fake
    for pred in disc_preds:
        loss += torch.mean(torch.pow(pred - target_val, 2))
    return loss / len(disc_preds)


# ==========================================
# DATASET
# ==========================================
class NpyDataset(Dataset):
    """Carica tensori .npy (3, 512, 512) da una cartella."""
    def __init__(self, folder):
        self.files = sorted(glob.glob(os.path.join(folder, '*.npy')))
        if not self.files:
            raise FileNotFoundError(f"Nessun file .npy trovato in: {folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        tensor = np.load(self.files[idx]).astype(np.float32)
        return torch.from_numpy(tensor)


class ReplayBuffer:
    """
    Buffer di 50 immagini fake storiche per stabilizzare il discriminatore.
    Con probabilità 50% restituisce un fake dal buffer invece dell'ultimo generato.
    """
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, images):
        result = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(img)
                result.append(img)
            elif random.random() > 0.5:
                idx = random.randint(0, self.max_size - 1)
                result.append(self.data[idx].clone())
                self.data[idx] = img
            else:
                result.append(img)
        return torch.cat(result, dim=0)


# ==========================================
# TRAINING
# ==========================================
class Tee:
    """Duplica stdout su terminale e file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def train_cyclegan(
    domain_a_dir,
    domain_b_dir,
    checkpoint_dir="checkpoints",
    num_epochs=200,
    batch_size=2,
    lr=2e-4,
    lambda_cycle=4.0,
    lambda_identity=0.2,
    lambda_trig=1.1,
    lambda_adv=1.2,   
    save_every=5,
    resume_path=None,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Modelli ---
    G_AB = Generator_2_1_2D().to(device)   # A -> B
    G_BA = Generator_2_1_2D().to(device)   # B -> A
    D_A = MultiScaleDiscriminator().to(device)
    D_B = MultiScaleDiscriminator().to(device)

    # --- Ottimizzatori ---
    opt_G = torch.optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=lr, betas=(0.5, 0.999)
    )
    opt_D = torch.optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()),
        lr=lr, betas=(0.5, 0.999)
    )

    start_epoch = 1

    # --- Resume da checkpoint ---
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        G_AB.load_state_dict(ckpt["G_AB"])
        G_BA.load_state_dict(ckpt["G_BA"])
        D_A.load_state_dict(ckpt["D_A"])
        D_B.load_state_dict(ckpt["D_B"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Ripreso da checkpoint epoch {ckpt['epoch']}")

    # --- LR Scheduler: costante per metà training, poi decay lineare a 0 ---
    decay_start = num_epochs // 2
    def lr_lambda(epoch):
        if epoch < decay_start:
            return 1.0
        return 1.0 - (epoch - decay_start) / (num_epochs - decay_start)
    
    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)
    # Avanza lo scheduler se si riprende da un checkpoint
    for _ in range(start_epoch - 1):
        sched_G.step()
        sched_D.step()

    # --- Replay Buffers ---
    buffer_A = ReplayBuffer(max_size=50)
    buffer_B = ReplayBuffer(max_size=50)

    # --- Dataset e DataLoader ---
    ds_A = NpyDataset(domain_a_dir)
    ds_B = NpyDataset(domain_b_dir)
    #loader_A = DataLoader(ds_A, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    #loader_B = DataLoader(ds_B, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    #usare questi per velocizzare il caricamento, ma attenzione alla RAM se i dataset sono grandi
    loader_A = DataLoader(ds_A, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True)
    loader_B = DataLoader(ds_B, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True)
    
    batches_per_epoch = min(len(loader_A), len(loader_B))

    print(f"Dominio A: {len(ds_A)} campioni  |  Dominio B: {len(ds_B)} campioni")
    print(f"Batch size: {batch_size}  |  Batch per epoch: {batches_per_epoch}")
    print(f"Epochs: {num_epochs}  |  lr: {lr}")
    print(f"Lambda cycle: {lambda_cycle}  |  Lambda identity: {lambda_identity}  |  Lambda trig: {lambda_trig}  |  Lambda adv: {lambda_adv}")
    print("=" * 90)

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        # Accumulatori per le medie di epoch
        acc = {k: 0.0 for k in [
            "G_adv", "G_cycle", "G_idt", "G_trig", "G_tot",
            "D_A", "D_B", "D_tot"
        ]}

        for batch_idx, (real_A, real_B) in enumerate(zip(loader_A, loader_B), 1):
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # ============================
            #  Aggiornamento Generatori
            # ============================
            G_AB.train(); G_BA.train()
            opt_G.zero_grad()

            # Forward
            fake_B = G_AB(real_A)       # A -> B
            fake_A = G_BA(real_B)       # B -> A
            rec_A = G_BA(fake_B)        # A -> B -> A (cycle)
            rec_B = G_AB(fake_A)        # B -> A -> B (cycle)

            # 1) Loss Avversaria (generatori vogliono ingannare i discriminatori)
            loss_G_adv_B = lsgan_loss(D_B(fake_B), is_real=True)
            loss_G_adv_A = lsgan_loss(D_A(fake_A), is_real=True)
            loss_G_adv = loss_G_adv_A + loss_G_adv_B

            # 2) Cycle Consistency Loss (pesata sulla magnitudo)
            loss_cycle_A = magnitude_aware_cycle_loss(rec_A, real_A)
            loss_cycle_B = magnitude_aware_cycle_loss(rec_B, real_B)
            loss_G_cycle = (loss_cycle_A + loss_cycle_B) * lambda_cycle

            # 3) Identity Loss (se dai B a G_AB, dovrebbe restituire B invariato)
            idt_B = G_AB(real_B)
            idt_A = G_BA(real_A)
            loss_idt = (
                magnitude_aware_cycle_loss(idt_B, real_B) +
                magnitude_aware_cycle_loss(idt_A, real_A)
            ) * lambda_identity

            # 4) Trigonometric Consistency Loss
            loss_trig = (
                trigonometric_consistency_loss(fake_B[:, 1:2], fake_B[:, 2:3]) +
                trigonometric_consistency_loss(fake_A[:, 1:2], fake_A[:, 2:3])
            ) * lambda_trig

            # Totale Generatori
            loss_G = (lambda_adv * loss_G_adv) + loss_G_cycle + loss_idt + loss_trig
            loss_G.backward()
            opt_G.step()

            # ============================
            #  Aggiornamento Discriminatori
            # ============================
            opt_D.zero_grad()

            # D_A: distinguere real_A da fake_A (usando il replay buffer)
            loss_D_A_real = lsgan_loss(D_A(real_A), is_real=True)
            loss_D_A_fake = lsgan_loss(D_A(buffer_A.push_and_pop(fake_A.detach())), is_real=False)
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

            # D_B: distinguere real_B da fake_B (usando il replay buffer)
            loss_D_B_real = lsgan_loss(D_B(real_B), is_real=True)
            loss_D_B_fake = lsgan_loss(D_B(buffer_B.push_and_pop(fake_B.detach())), is_real=False)
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

            loss_D = loss_D_A + loss_D_B
            # Update D less frequently
            if batch_idx % 2 == 0:  # Update D every 2nd batch
                loss_D.backward()
                opt_D.step()

            # --- Accumula per media epoch ---
            acc["G_adv"]   += loss_G_adv.item()
            acc["G_cycle"] += loss_G_cycle.item()
            acc["G_idt"]   += loss_idt.item()
            acc["G_trig"]  += loss_trig.item()
            acc["G_tot"]   += loss_G.item()
            acc["D_A"]     += loss_D_A.item()
            acc["D_B"]     += loss_D_B.item()
            acc["D_tot"]   += loss_D.item()

        # --- Riepilogo Epoch ---
        elapsed = time.time() - epoch_start
        current_lr = opt_G.param_groups[0]['lr']
        print("-" * 90)
        print(f">>> Epoch {epoch}/{num_epochs} completata in {elapsed:.1f}s  |  LR: {current_lr:.6f}  |  Medie:")
        for k, v in acc.items():
            print(f"    {k:10s}: {v / batches_per_epoch:.4f}")
        print("=" * 90)

        # --- Aggiornamento Learning Rate ---
        sched_G.step()
        sched_D.step()

        # --- Salvataggio checkpoint ogni N epoch ---
        if epoch % save_every == 0 or epoch == num_epochs:
            torch.save({
                "epoch": epoch,
                "G_AB": G_AB.state_dict(),
                "G_BA": G_BA.state_dict(),
                "D_A": D_A.state_dict(),
                "D_B": D_B.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
            }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pt"))
            print(f"    Checkpoint salvato: checkpoint_epoch_{epoch:03d}.pt")

    print("Training completato!")


# --- ESECUZIONE ---
if __name__ == "__main__":
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_path = os.path.join(checkpoint_dir, "training_log.txt")

    with open(log_path, "a", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_file)):
            print("\n" + "=" * 90)
            print(f"Log training: {log_path}")
            print("=" * 90)
            train_cyclegan(
                domain_a_dir="dataset_dominio_A",
                domain_b_dir="dataset_dominio_B",
                checkpoint_dir=checkpoint_dir,
                num_epochs=200,
                batch_size=2,
                resume_path=os.path.join("checkpoints", "checkpoint_epoch_130.pt"),
            )


