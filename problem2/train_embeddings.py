

import os
import re
import sys
import json
import time
import argparse
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn



def clean_text(text):
    # Convert to lowercase
    # Remove non-alphabetic characters except spaces
    # Split into words
    # Remove very short words (< 2 characters)
    t = (text or "").lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    words = [w for w in t.split() if len(w) >= 2]
    return words


class TextAutoencoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_dim):
        super().__init__()
        # Encoder: vocab_size → hidden_dim → embedding_dim
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        # Decoder: embedding_dim → hidden_dim → vocab_size
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
            nn.Sigmoid()  # Output probabilities
        )

    def forward(self, x):
        # Encode to bottleneck
        embedding = self.encoder(x)
        # Decode back to vocabulary space
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding




def set_seed(seed: int = 42):
    torch.manual_seed(seed)


def load_papers(input_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("Error: input JSON must be a list.", file=sys.stderr)
        sys.exit(1)

    papers = []
    for p in data:
        pid = p.get("arxiv_id") or p.get("id") or ""
        if not pid:
            continue
        abstract = p.get("abstract") or p.get("summary") or ""
        papers.append({"arxiv_id": pid, "abstract": abstract})
    if not papers:
        print("Error: no valid papers with arxiv_id found.", file=sys.stderr)
        sys.exit(1)
    return papers


def build_vocab(abstracts, max_vocab: int):
    cnt = Counter()
    for a in abstracts:
        cnt.update(clean_text(a))
    keep = max(1, max_vocab - 1)  
    most = cnt.most_common(keep)
    vocab_to_idx = {w: i + 1 for i, (w, _) in enumerate(most)}
    return vocab_to_idx  


def bow_vector(words, vocab_to_idx, vocab_size: int, binary: bool = True):

    x = torch.zeros(vocab_size, dtype=torch.float32)
    if binary:
        for w in words:
            idx = vocab_to_idx.get(w, 0)
            if idx > 0:
                x[idx] = 1.0
    else:
        for w in words:
            idx = vocab_to_idx.get(w, 0)
            if idx > 0:
                x[idx] += 1.0
        if x.max() > 0:
            x = x / x.max()
    return x


def make_dataset(papers, vocab_to_idx, vocab_size: int, seq_len: int):
    """
    Turn each paper into (arxiv_id, BoW tensor).
    We cap tokens used per abstract by seq_len (for speed/consistency).
    """
    dataset = []
    for p in papers:
        pid = p["arxiv_id"]
        words = clean_text(p["abstract"])[:seq_len]
        x = bow_vector(words, vocab_to_idx, vocab_size, binary=True)
        dataset.append((pid, x))
    return dataset


def batch_iter(dataset, batch_size: int):
    """Simple python batching (CPU-friendly)."""
    for i in range(0, len(dataset), batch_size):
        chunk = dataset[i:i + batch_size]
        ids = [pid for pid, _ in chunk]
        xb = torch.stack([x for _, x in chunk])  # [B, V]
        yield ids, xb


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)




def parse_args():
    ap = argparse.ArgumentParser(description="Train BoW Autoencoder for embeddings")
    ap.add_argument("input_json", type=str, help="path to papers.json")
    ap.add_argument("output_dir", type=str, help="directory to save outputs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--embedding_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--max_vocab", type=int, default=5000)
    ap.add_argument("--seq_len", type=int, default=150, help="max tokens used per abstract")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)


    papers = load_papers(args.input_json)
    abstracts = [p["abstract"] for p in papers]
    print(f"Loaded {len(papers)} papers")


    vocab_to_idx = build_vocab(abstracts, args.max_vocab)
    V = len(vocab_to_idx) + 1 
    print(f"Vocab size (including UNK): {V}")

    dataset = make_dataset(papers, vocab_to_idx, V, args.seq_len)


    model = TextAutoencoder(vocab_size=V, hidden_dim=args.hidden_dim, embedding_dim=args.embedding_dim)
    total_params = count_params(model)
    print(f"Total parameters: {total_params}")
    if total_params >= 2_000_000:
        print("Parameter limit exceeded (>= 2,000,000). Reduce vocab/hidden/embedding.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cpu")  
    model.to(device)

 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    start_ts = time.time()
    epoch_losses = []

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for ids, xb in batch_iter(dataset, args.batch_size):
            xb = xb.to(device)          
            recon, _ = model(xb)        
            loss = criterion(recon, xb)  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = sum(losses) / max(1, len(losses))
        epoch_losses.append(avg_loss)
        print(f"Epoch {ep:03d}/{args.epochs}  Loss: {avg_loss:.4f}")

    end_ts = time.time()


    model_path = os.path.join(args.output_dir, "model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_to_idx": vocab_to_idx,
        "model_config": {
            "vocab_size": V,
            "hidden_dim": args.hidden_dim,
            "embedding_dim": args.embedding_dim
        }
    }, model_path)
    print(f"Saved: {model_path}")


    model.eval()
    out = []
    with torch.no_grad():
        for pid, x in dataset:
            x = x.unsqueeze(0).to(device)    # [1, V]
            recon, emb = model(x)
            rec_loss = criterion(recon, x).item()
            out.append({
                "arxiv_id": pid,
                "embedding": emb.squeeze(0).cpu().tolist(),
                "reconstruction_loss": float(rec_loss)
            })
    emb_path = os.path.join(args.output_dir, "embeddings.json")
    save_json(out, emb_path)
    print(f"Saved: {emb_path}")


    vocab_json = {
        "vocab_to_idx": vocab_to_idx,
        "idx_to_vocab": {str(i): w for w, i in vocab_to_idx.items()},
        "vocab_size": V
    }
    vocab_path = os.path.join(args.output_dir, "vocabulary.json")
    save_json(vocab_json, vocab_path)
    print(f"Saved: {vocab_path}")

    log = {
        "start_time": datetime.fromtimestamp(start_ts).isoformat(),
        "end_time": datetime.fromtimestamp(end_ts).isoformat(),
        "epochs": args.epochs,
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "loss_curve": epoch_losses,
        "total_parameters": total_params,
        "papers_processed": len(dataset),
        "embedding_dimension": args.embedding_dim,
        "hidden_dimension": args.hidden_dim,
        "vocab_size": V,
        "max_vocab": args.max_vocab,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "seed": args.seed,
        "input_file": os.path.abspath(args.input_json),
    }
    log_path = os.path.join(args.output_dir, "training_log.json")
    save_json(log, log_path)
    print(f"Saved: {log_path}")
    print("Done.")


if __name__ == "__main__":
    main()
