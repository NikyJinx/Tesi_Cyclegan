import math
import os
import argparse
import numpy as np
import librosa
import torch

MODEL_PATH = r"C:\CycleGan\STFT\genre_classifier_wav\genre_classifier_wav_best_jit.pt"
INPUT_PATH = r"C:\CycleGan\STFT\TEST"

SR = 22050
SEGMENT_SECONDS = 25.0
HOP_SECONDS = 1.0
PRE_EMPHASIS = 0.0
SUPPORTED_EXTENSIONS = (".wav", ".waw")

def build_chunks(y, sr, segment_seconds, hop_seconds, pre_emphasis=0.0):
    segment_samples = int(sr * segment_seconds)
    hop_samples = int(sr * hop_seconds)

    chunks = []
    if len(y) < segment_samples:
        y = np.pad(y, (0, segment_samples - len(y)), mode="constant")

    n = int(math.floor((len(y) - segment_samples) / hop_samples)) + 1
    for i in range(n):
        start = i * hop_samples
        chunk = y[start:start + segment_samples]

        if len(chunk) < segment_samples:
            chunk = np.pad(chunk, (0, segment_samples - len(chunk)), mode="constant")

        if pre_emphasis > 0.0:
            chunk = np.append(chunk[0], chunk[1:] - pre_emphasis * chunk[:-1])

        peak = np.max(np.abs(chunk))
        if peak > 1e-9:
            chunk = chunk / peak

        chunks.append(chunk.astype(np.float32))

    return np.stack(chunks, axis=0)


def predict_file(model, device, wav_path, sr, segment_seconds, hop_seconds, pre_emphasis):
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    x = build_chunks(y, sr, segment_seconds, hop_seconds, pre_emphasis)

    with torch.no_grad():
        x_t = torch.from_numpy(x).to(device)
        logits = model(x_t)
        if logits.ndim == 2 and logits.shape[1] == 2:
            # Modello binario con 2 classi esplicite: [A, B]
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_a = probs[:, 0]
            probs_b = probs[:, 1]
            prob_mode = "softmax_2class"
        else:
            # Modello con un solo logit (probabilita di B): A e ottenuto come complemento.
            probs_b = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            probs_a = 1.0 - probs_b
            prob_mode = "sigmoid_1logit"

    mean_prob_b = float(np.mean(probs_b))
    median_prob_b = float(np.median(probs_b))
    std_prob_b = float(np.std(probs_b))
    final_score_b = 0.6 * mean_prob_b + 0.4 * median_prob_b

    mean_prob_a = float(np.mean(probs_a))
    median_prob_a = float(np.median(probs_a))
    std_prob_a = float(np.std(probs_a))
    final_score_a = 0.6 * mean_prob_a + 0.4 * median_prob_a

    audio_seconds = len(y) / float(sr)

    return {
        "path": wav_path,
        "duration": audio_seconds,
        "chunks": len(probs_b),
        "prob_mode": prob_mode,
        "mean_a": mean_prob_a,
        "median_a": median_prob_a,
        "std_a": std_prob_a,
        "score_a": final_score_a,
        "probs_a": probs_a,
        "mean_b": mean_prob_b,
        "median_b": median_prob_b,
        "std_b": std_prob_b,
        "score_b": final_score_b,
        "probs_b": probs_b,
    }


def list_audio_files(input_path):
    if os.path.isdir(input_path):
        files = sorted(
            os.path.join(input_path, name)
            for name in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, name))
            and os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS
        )
        return files

    ext = os.path.splitext(input_path)[1].lower()
    if os.path.isfile(input_path) and ext in SUPPORTED_EXTENSIONS:
        return [input_path]

    return []

def main():
    parser = argparse.ArgumentParser(description="Test classificatore genere su audio con chunk temporali")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument(
        "--input_path",
        type=str,
        default=INPUT_PATH,
        help="File audio singolo o cartella contenente file .wav/.waw",
    )
    parser.add_argument("--sr", type=int, default=SR)
    parser.add_argument("--segment_seconds", type=float, default=SEGMENT_SECONDS)
    parser.add_argument("--hop_seconds", type=float, default=HOP_SECONDS)
    parser.add_argument("--pre_emphasis", type=float, default=PRE_EMPHASIS)
    parser.add_argument("--decision_threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(args.model_path, map_location=device)
    model.eval()

    audio_files = list_audio_files(args.input_path)
    if not audio_files:
        raise RuntimeError(
            f"Nessun file audio valido trovato in: {args.input_path} "
            f"(estensioni supportate: {', '.join(SUPPORTED_EXTENSIONS)})"
        )

    print(f"Input: {args.input_path}")
    print(f"File trovati: {len(audio_files)}")
    print(f"Modello: {args.model_path}")

    num_a = 0
    num_b = 0
    scores = []

    for idx, wav_path in enumerate(audio_files, start=1):
        result = predict_file(
            model=model,
            device=device,
            wav_path=wav_path,
            sr=args.sr,
            segment_seconds=args.segment_seconds,
            hop_seconds=args.hop_seconds,
            pre_emphasis=args.pre_emphasis,
        )

        label = "dominio B" if result["score_b"] >= args.decision_threshold else "dominio A"
        if label == "dominio B":
            num_b += 1
        else:
            num_a += 1
        scores.append(result["score_b"])

        print(f"\n[{idx}/{len(audio_files)}] Audio: {wav_path}")
        print(f"Durata audio (s): {result['duration']:.2f}")
        if result["prob_mode"] == "softmax_2class":
            print("Modalita probabilita: softmax 2 classi (A/B reali)")
        else:
            print("Modalita probabilita: sigmoid 1 logit (A = 1 - B)")
        print(f"Parametri chunk -> segment={args.segment_seconds:.2f}s, hop={args.hop_seconds:.2f}s")
        print(f"Chunk analizzati: {result['chunks']}")
        print(f"Probabilità media dominio A: {result['mean_a']:.4f}")
        print(f"Probabilità mediana dominio A: {result['median_a']:.4f}")
        print(f"Score finale robusto dominio A: {result['score_a']:.4f}")
        print(f"Deviazione standard dominio A: {result['std_a']:.4f}")
        print(f"Probabilità media dominio B: {result['mean_b']:.4f}")
        print(f"Probabilità mediana dominio B: {result['median_b']:.4f}")
        print(f"Score finale robusto dominio B: {result['score_b']:.4f}")
        print(f"Deviazione standard dominio B: {result['std_b']:.4f}")
        print(f"Classificazione finale: {label}")
        print("Prime 10 probabilità chunk dominio A:", result["probs_a"][:10])
        print("Prime 10 probabilità chunk dominio B:", result["probs_b"][:10])

    print("\n=== RIEPILOGO ===")
    print(f"Totale file analizzati: {len(audio_files)}")
    print(f"Classificati dominio A: {num_a}")
    print(f"Classificati dominio B: {num_b}")
    print(f"Score medio dataset: {float(np.mean(scores)):.4f}")

if __name__ == "__main__":
    main()