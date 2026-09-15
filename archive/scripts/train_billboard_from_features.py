#!/usr/bin/env python3
"""Billboard training using pre-extracted McGill chroma features."""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.data.billboard_loader import BillboardDataset

QUALITY5 = ["maj", "min", "dom", "hdim", "dim"]
CACHE_DIR = REPO / "data" / "cache" / "billboard"
MODELS_DIR = REPO / "data" / "models"
CORPUS_FILE = CACHE_DIR / "billboard_training_corpus_v2.npz"
QUALITY_MODEL_FILE = MODELS_DIR / "billboard_quality_head_v2.pt"
ROOT_MODEL_FILE = MODELS_DIR / "billboard_root_model_v2.npz"
REPORT_FILE = REPO / "docs" / "billboard_training_results_v2.md"

MCGILL_BILLBOARD_PATH = Path.home() / "mir_datasets" / "billboard" / "McGill-Billboard"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_mcgill_features(song_id: str) -> dict:
    """Load pre-extracted chroma features from McGill Billboard.

    Format: audio_path, time, chroma1_0...chroma1_11, chroma2_0...chroma2_11
    """
    song_num = f"{int(song_id):04d}"
    song_path = MCGILL_BILLBOARD_PATH / song_num

    if not song_path.exists():
        return None

    chroma_file = song_path / "bothchroma.csv"
    if not chroma_file.exists():
        return None

    try:
        # Read raw data - format: path, time, 12 chroma values (x2 for "both")
        data = []
        times = []
        with open(chroma_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 14:
                    continue
                # Skip audio path (parts[0]), extract time and chroma values
                time_val = float(parts[1])
                times.append(time_val)
                # Take first 12 chroma values (skip the second set)
                chroma_vals = [float(x) for x in parts[2:14]]
                data.append(chroma_vals)

        if not data:
            return None

        # Convert to (12, T) matrix
        chroma = np.array(data, dtype=np.float32).T

        return {
            "chroma": chroma,
            "times": np.array(times, dtype=np.float32),
            "track_id": song_id,
        }
    except Exception as e:
        return None


def extract_billboard_features_from_mcgill(force: bool = False):
    """Extract training data from McGill Billboard chroma."""
    print(f"\n{'=' * 70}")
    print(f"PHASE 1: Extract Features from McGill Billboard Chroma")
    print(f"{'=' * 70}\n")

    if CORPUS_FILE.exists() and not force:
        print(f"✓ Corpus already exists: {CORPUS_FILE}")
        return CORPUS_FILE

    bb = BillboardDataset(chord_type="majmin")
    bb_mirdata = bb.dataset
    all_song_ids = list(bb_mirdata.track_ids)

    print(f"Processing {len(all_song_ids)} Billboard songs")

    all_feats = []
    all_roots = []
    all_qualities = []
    all_song_ids_used = []
    all_metadata = []
    failures = []

    for idx, song_id in enumerate(all_song_ids):
        try:
            mcgill = load_mcgill_features(song_id)
            if mcgill is None:
                failures.append((song_id, "mcgill_features_missing"))
                continue

            chroma = mcgill["chroma"]
            times = mcgill["times"]

            gt = bb.load_track_gt(song_id)
            if not gt or not gt["chords"]:
                failures.append((song_id, "no_chords"))
                continue

            for chord in gt["chords"]:
                t0, t1 = chord["t0"], chord["t1"]
                root = chord["root"]
                quality = chord["quality"]
                label = chord["label"]

                frame_mask = (times >= t0) & (times < t1)
                if not np.any(frame_mask):
                    continue

                feat12 = np.mean(chroma[:, frame_mask], axis=1)

                all_feats.append(feat12)
                all_roots.append(root)
                all_qualities.append(quality)
                all_song_ids_used.append(song_id)
                all_metadata.append({"t0": t0, "t1": t1, "label": label})

            if (idx + 1) % 100 == 0 or idx == len(all_song_ids) - 1:
                print(f"[{idx+1}/{len(all_song_ids)}] ✓ {song_id}")

        except Exception as e:
            failures.append((song_id, str(e)))

    print(f"\nExtracted {len(all_feats)} chord events from {len(set(all_song_ids_used))} songs")
    print(f"Failures: {len(failures)}")

    if len(all_feats) == 0:
        raise RuntimeError("No features extracted!")

    q5_map = {q: i for i, q in enumerate(QUALITY5)}
    q5_map[None] = -1
    quality_idx = np.array([q5_map.get(q, -1) for q in all_qualities], dtype=int)

    np.savez_compressed(
        CORPUS_FILE,
        feats=np.array(all_feats, dtype=np.float32),
        roots=np.array(all_roots, dtype=object),
        quality_idx=quality_idx,
        song_id=np.array(all_song_ids_used, dtype=object),
        qualities=np.array(QUALITY5, dtype=object),
        metadata=all_metadata,
    )

    print(f"✓ Saved corpus: {CORPUS_FILE}\n")
    return CORPUS_FILE


class QualityHeadMLP(nn.Module):
    def __init__(self, in_dim: int = 12, n_classes: int = 5, h1: int = 64, h2: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(h2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_quality_model(epochs: int = 100, batch_size: int = 32, learning_rate: float = 1e-3):
    """Train 5-way quality head."""
    print(f"\n{'=' * 70}")
    print(f"PHASE 2: Train Quality Head (maj/min/dom/hdim/dim)")
    print(f"{'=' * 70}\n")

    d = np.load(CORPUS_FILE, allow_pickle=True)
    feats = d["feats"].astype(np.float32)
    quality_idx = d["quality_idx"].astype(int)
    song_id = d["song_id"]

    keep = quality_idx >= 0
    X = feats[keep]
    y = quality_idx[keep]
    songs = song_id[keep]

    print(f"Clean 5-way training examples: {len(X)}")
    print(f"Class distribution: {np.bincount(y)}")

    unique_songs = np.unique(songs)
    np.random.seed(42)
    np.random.shuffle(unique_songs)

    n_train = int(0.8 * len(unique_songs))
    train_songs = set(unique_songs[:n_train])
    val_songs = set(unique_songs[n_train:])

    train_mask = np.isin(songs, list(train_songs))
    val_mask = np.isin(songs, list(val_songs))

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"Train: {len(X_train)} samples | Val: {len(X_val)} samples\n")

    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QualityHeadMLP(in_dim=X_train.shape[1], n_classes=5).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_norm, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val_norm, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    best_val_acc = 0
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
                val_correct += (logits.argmax(dim=1) == y_batch).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / len(X_val)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | train_loss={train_loss:.4f} | val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), QUALITY_MODEL_FILE)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step()

    model.load_state_dict(torch.load(QUALITY_MODEL_FILE, map_location=device))
    print(f"\n✓ Saved: {QUALITY_MODEL_FILE}")
    print(f"Best val acc: {best_val_acc:.3f}")

    return model, X_val_norm, y_val


def train_root_model():
    """Train 12-way root classifier."""
    print(f"\n{'=' * 70}")
    print(f"PHASE 3: Train Root Classifier (12 roots)")
    print(f"{'=' * 70}\n")

    d = np.load(CORPUS_FILE, allow_pickle=True)
    feats = d["feats"].astype(np.float32)
    roots = d["roots"]
    song_id = d["song_id"]
    quality_idx = d["quality_idx"].astype(int)

    keep = quality_idx >= 0
    X = feats[keep]
    y_roots = roots[keep]
    songs = song_id[keep]

    # Roots are already stored as integers 0-11
    y = np.array([r if isinstance(r, int) else -1 for r in y_roots], dtype=int)
    keep_valid = y >= 0

    X = X[keep_valid]
    y = y[keep_valid]
    songs = songs[keep_valid]

    print(f"Training examples: {len(X)}")
    print(f"Class distribution: {np.bincount(y)}")

    unique_songs = np.unique(songs)
    np.random.seed(42)
    np.random.shuffle(unique_songs)

    n_train = int(0.8 * len(unique_songs))
    train_songs = set(unique_songs[:n_train])
    val_songs = set(unique_songs[n_train:])

    train_mask = np.isin(songs, list(train_songs))
    val_mask = np.isin(songs, list(val_songs))

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"Train: {len(X_train)} samples | Val: {len(X_val)} samples\n")

    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)

    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    )

    print("Training logistic regression...")
    model.fit(X_train_norm, y_train)

    train_acc = model.score(X_train_norm, y_train)
    val_acc = model.score(X_val_norm, y_val)

    print(f"Train acc: {train_acc:.3f}")
    print(f"Val acc: {val_acc:.3f}\n")

    np.savez_compressed(
        ROOT_MODEL_FILE,
        coef=model.coef_,
        intercept=model.intercept_,
        classes=model.classes_,
    )

    print(f"✓ Saved: {ROOT_MODEL_FILE}")

    return model, X_val_norm, y_val


def evaluate(quality_model, root_model, X_val_quality, y_val_quality, X_val_root, y_val_root):
    """Evaluate models."""
    print(f"\n{'=' * 70}")
    print(f"PHASE 4: Evaluation")
    print(f"{'=' * 70}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    quality_model.eval()
    with torch.no_grad():
        logits = quality_model(torch.tensor(X_val_quality, dtype=torch.float32).to(device))
        quality_pred = logits.argmax(dim=1).cpu().numpy()

    quality_acc = (quality_pred == y_val_quality).mean()
    print(f"Quality accuracy: {quality_acc:.3f}")

    root_pred = root_model.predict(X_val_root)
    root_acc = (root_pred == y_val_root).mean()
    print(f"Root accuracy: {root_acc:.3f}")

    baseline = {"root": 0.59, "majmin": 0.61}
    print(f"\nVs iRealb: quality={quality_acc:.1%} vs {baseline['majmin']:.1%} ({(quality_acc - baseline['majmin'])*100:+.1f}pp)")

    return {
        "quality_acc": float(quality_acc),
        "root_acc": float(root_acc),
        "delta_vs_baseline": float(quality_acc - baseline['majmin']),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.all or not (args.extract or args.train or args.evaluate):
        args.extract = args.train = args.evaluate = True

    try:
        if args.extract:
            extract_billboard_features_from_mcgill(force=args.force)

        quality_model = None
        root_model = None
        if args.train:
            quality_model, X_val_quality, y_val_quality = train_quality_model()
            root_model, X_val_root, y_val_root = train_root_model()

        if args.evaluate and quality_model and root_model:
            results = evaluate(quality_model, root_model, X_val_quality, y_val_quality, X_val_root, y_val_root)
            print(f"\nResults:\n{json.dumps(results, indent=2)}")

            with open(REPORT_FILE, "w") as f:
                f.write(f"# Billboard McGill Model Training (v2)\n\n")
                f.write(f"Quality accuracy: {results['quality_acc']:.1%}\n")
                f.write(f"Root accuracy: {results['root_acc']:.1%}\n")
                f.write(f"Delta vs iRealb: {results['delta_vs_baseline']:+.1%}pp\n")

            print(f"\n✓ Report: {REPORT_FILE}")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
