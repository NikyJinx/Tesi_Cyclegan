# Tesi CycleGAN: Audio Style Transfer

Questa repository contiene il codice sviluppato per il progetto di tesi magistrale Audio Style Transfer. 
Il sistema utilizza una rete generativa avversaria (CycleGAN) per mappare l'audio da un dominio a un altro (es. Violin $\leftrightarrow$ Piano) operando direttamente sulle rappresentazioni a spettrogramma (STFT) del segnale acustico.

**Note:** 
I file presenti nella directory principale rappresentano le **versioni finali** del codice (CycleGAN V6 Modified). Eventuali sottocartelle contengono esclusivamente esperimenti precedenti, script di test e non sono necessarie per l'esecuzione del progetto.

---

##  Architettura Finale: CycleGAN V6 Modified

L'implementazione finale (`Cyclegan_V6_modified.py`) introduce modifiche architetturali specifiche per il dominio audio rispetto alle GAN tradizionali per immagini:

* **Kernel Asimmetrici e Rettangolari:** Nei blocchi residui (`ResidualBlock2D_GLU`) vengono estratti kernel di dimensione `(3, 7)` e `(1, 9)` per modellare in modo indipendente la risoluzione temporale e frequenziale.
* **Dilatazione Temporale 1D:** L'espansione del campo ricettivo avviene in modo esclusivo sull'asse orizzontale del tempo (dilations: 1, 2, 4, 8, 1, 2) per catturare le dipendenze temporali lunghe senza distorcere le frequenze.
* **Loss Avanzate per l'Audio:**
  * `magnitude_aware_cycle_loss`: Calcola la cycle consistency loss mascherando gli errori di fase nelle zone di silenzio, utilizzando la magnitudo reale come peso.
  * `trigonometric_consistency_loss`: Forza i canali che rappresentano il seno e il coseno della fase a giacere sul cerchio unitario ($sin^2\theta + cos^2\theta = 1$).

---

## Componenti Principali del Repository

L'intero flusso di lavoro è gestito dagli script presenti nella directory root:

### 1. Preprocessing dei Dati
* **Creazione Dataset NPY:** Lo script dedicato converte i file audio (WAV, MP3, ecc.) in tensori `.npy` con dtype `float32`. Genera tensori a 3 canali `[Magnitudo, Seno, Coseno]` con shape `(3, 512, 2048)`. La magnitudo subisce una normalizzazione lineare nel range `[-1, 1]` ottimizzata per le attivazioni Tanh del generatore.
* **Esportazione STFT in Immagini (TIFF/PNG):** Un modulo di utility permette di estrarre le finestre STFT e i Mel-spectrogrammi salvandoli come immagini (Amplitude, Phase, e 3-layer composito) per l'ispezione visiva dei dati processati.

### 2. Modello e Addestramento (`Cyclegan_V6_modified.py`)
Contiene l'architettura completa (Generatore `Generator_2_1_2D`, Discriminatore Multi-Scala `MultiScaleDiscriminator`) e il training loop. Supporta il caricamento di un replay buffer per i file "fake" e lo scheduling del learning rate.

### 3. Inferenza e Ricostruzione Audio
Lo script di inferenza carica un checkpoint addestrato (es. `G_AB` o `G_BA`), divide un nuovo file audio in frammenti, ne prevede la trasformazione applicando il modello e fonde i blocchi risultanti. Ricostruisce infine l'onda audio nel dominio del tempo tramite `librosa.istft`, con l'opzione di usare la fase generata o forzare la fase originale dell'input.

### 4. Classificatore di Genere per Valutazione
Per valutare in modo oggettivo l'efficacia del trasferimento di stile, il progetto include un classificatore binario basato su CNN 1D (`GenreWaveformCNN`):
* **Training:** Addestra una rete che lavora direttamente sulla forma d'onda grezza (raw waveform) per distinguere i due domini musicali (Domain A vs Domain B).
* **Inference/Test:** Suddivide i file convertiti in segmenti temporali e restituisce uno score aggregato (media e mediana delle probabilità) tramite una classificazione softmax a 2 logit, determinando a quale dominio appartiene il file audio finale.

---
## Configurazione dell'Ambiente Virtuale (Conda)

Per facilitare l'esecuzione degli script e garantire la compatibilità di tutte le librerie, la repo include il file `environment.yml`, che permette di ricreare esattamente l'ambiente virtuale Conda utilizzato durante lo sviluppo.

Segui questi passaggi per configurare l'ambiente:

1. **Clona il repository:**
   Scarica il progetto in locale tramite Git e accedi alla cartella.
   ```bash
   git clone https://github.com/NikyJinx/Tesi_Cyclegan.git
   cd Tesi_Cyclegan
   ```

2. **Modifica il percorso nel file environment:**
   Apri il file `environment.yml` con un editor di testo. Vai all'ultima riga del file che inizia con `prefix:` e modificala inserendo il percorso di installazione di Anaconda del tuo computer. In alternativa puoi eliminare completamente l'intera riga per lasciare che Conda installi l'ambiente nel suo percorso predefinito.

3. **Crea l'ambiente da file:**
   Usa il comando Conda per leggere il file YAML e installare tutte le dipendenze necessarie.
   ```bash
   conda env create -f environment.yml
   ```

4. **Attiva l'ambiente:**
   Una volta terminata l'installazione, attiva l'ambiente virtuale.
   ```bash
   conda activate bigvgan
   ```
   *(Nota bene: il nome dell'ambiente è bigvgan come specificato alla prima riga all'interno del file environment.yml)*



## Requisiti di Sistema

* Python 3.8+ (Version used: 3.10.19)
* PyTorch (raccomandato supporto CUDA)
* Librosa e Soundfile (per l'elaborazione del segnale acustico)
* NumPy
* Tifffile (per l'esportazione degli spettrogrammi in alta qualità)
