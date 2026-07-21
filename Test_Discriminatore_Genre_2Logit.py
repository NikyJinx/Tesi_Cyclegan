import math
import os
import argparse
import numpy as np
import librosa
import torch

MODEL_PATH = r"C:\CycleGan\STFT\genre_classifier_wav\genre_classifier_wav2logit_best_jit.pt"
INPUT_PATH = r"C:\CycleGan\STFT\TEST"

SR = 22050
SEGMENT_SECONDS = 10.0
HOP_SECONDS = 1.0
PRE_EMPHASIS = 0.0
SUPPORTED_EXTENSIONS = (".wav", ".waw", ".mp3", ".opus")


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


def predict_file(model, device, wav_path, sr, segment_seconds, hop_seconds, pre_emphasis, batch_size=16):
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    x = build_chunks(y, sr, segment_seconds, hop_seconds, pre_emphasis)

    probs_list_a = []
    probs_list_b = []

    with torch.no_grad():
        # IL SEGRETO E' QUI: Processiamo i frammenti a blocchi di 16 alla volta
        # Invece di inviare 216 frammenti tutti insieme, non saturiamo la memoria
        for i in range(0, len(x), batch_size):
            x_batch = x[i : i + batch_size]
            x_t = torch.from_numpy(x_batch).to(device)
            
            logits = model(x_t)  # shape (B, 2)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            probs_list_a.extend(probs[:, 0])
            probs_list_b.extend(probs[:, 1])

    probs_a = np.array(probs_list_a)
    probs_b = np.array(probs_list_b)

    audio_seconds = len(y) / float(sr)

    return {
        "path": wav_path,
        "duration": audio_seconds,
        "chunks": len(probs_a),
        "mean_a": float(np.mean(probs_a)),
        "median_a": float(np.median(probs_a)),
        "std_a": float(np.std(probs_a)),
        "score_a": 0.6 * float(np.mean(probs_a)) + 0.4 * float(np.median(probs_a)),
        "probs_a": probs_a,
        "mean_b": float(np.mean(probs_b)),
        "median_b": float(np.median(probs_b)),
        "std_b": float(np.std(probs_b)),
        "score_b": 0.6 * float(np.mean(probs_b)) + 0.4 * float(np.median(probs_b)),
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
    parser = argparse.ArgumentParser(
        description="Test classificatore genere 2-logit (softmax A/B) su audio con chunk temporali"
    )
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument(
        "--input_path",
        type=str,
        default=INPUT_PATH,
        help="File audio singolo o cartella contenente file audio",
    )
    parser.add_argument("--sr", type=int, default=SR)
    parser.add_argument("--segment_seconds", type=float, default=SEGMENT_SECONDS)
    parser.add_argument("--hop_seconds", type=float, default=HOP_SECONDS)
    parser.add_argument("--pre_emphasis", type=float, default=PRE_EMPHASIS)
    parser.add_argument("--decision_threshold", type=float, default=0.5,
                        help="Soglia su score_b per classificare come dominio B")
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

    print(f"Input:        {args.input_path}")
    print(f"File trovati: {len(audio_files)}")
    print(f"Modello:      {args.model_path}")
    print(f"Modalita:     softmax 2 logit (P(A) e P(B) indipendenti)")

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

        print(f"\n[{idx}/{len(audio_files)}] {wav_path}")
        print(f"  Durata (s):              {result['duration']:.2f}")
        print(f"  Chunk analizzati:        {result['chunks']}")
        print(f"  Parametri chunk:         segment={args.segment_seconds:.1f}s  hop={args.hop_seconds:.1f}s")
        print(f"  --- Dominio A ---")
        print(f"    media P(A):            {result['mean_a']:.4f}")
        print(f"    mediana P(A):          {result['median_a']:.4f}")
        print(f"    std P(A):              {result['std_a']:.4f}")
        print(f"    score robusto A:       {result['score_a']:.4f}")
        print(f"  --- Dominio B ---")
        print(f"    media P(B):            {result['mean_b']:.4f}")
        print(f"    mediana P(B):          {result['median_b']:.4f}")
        print(f"    std P(B):              {result['std_b']:.4f}")
        print(f"    score robusto B:       {result['score_b']:.4f}")
        print(f"  Classificazione finale:  {label}")
        print(f"  Prime 10 P(A) per chunk: {result['probs_a'][:10]}")
        print(f"  Prime 10 P(B) per chunk: {result['probs_b'][:10]}")

    print("\n=== RIEPILOGO ===")
    print(f"Totale file analizzati: {len(audio_files)}")
    print(f"Classificati dominio A: {num_a}")
    print(f"Classificati dominio B: {num_b}")
    print(f"Score medio dataset:    {float(np.mean(scores)):.4f}")


if __name__ == "__main__":
    main()
