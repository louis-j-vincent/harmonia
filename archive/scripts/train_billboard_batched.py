#!/usr/bin/env python3
"""Billboard training with disk-aware batch processing.

This script downloads Billboard audio in batches, extracts features,
immediately deletes audio, and trains models from cached features.
Never keeps more than one batch of audio on disk.
"""

import os
import sys
import json
import shutil
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.data.billboard_loader import BillboardDataset
from harmonia.models.stage1_pitch import PitchExtractor

# Configuration
QUALITY5 = ["maj", "min", "dom", "hdim", "dim"]
CACHE_DIR = REPO / "data" / "cache" / "billboard"
MODELS_DIR = REPO / "data" / "models"
BATCH_CACHE_DIR = CACHE_DIR / "batches"
CORPUS_FILE = CACHE_DIR / "billboard_training_corpus.npz"
ROOT_MODEL_FILE = MODELS_DIR / "billboard_root_model_v1.pt"
QUALITY_MODEL_FILE = MODELS_DIR / "billboard_quality_head_v1.pt"
REPORT_FILE = REPO / "docs" / "billboard_training_results.md"

BATCH_SIZE = 30  # Songs per batch (conservative for disk)
DISK_CRITICAL_THRESHOLD = 2.0  # GB - abort if lower than this

CACHE_DIR.mkdir(parents=True, exist_ok=True)
BATCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── Disk Monitoring ───────────────────────────────────────────────────────────

def get_disk_usage(path: str = "/") -> dict:
    """Get disk usage stats (macOS compatible)."""
    try:
        result = os.popen(f"df {path} | tail -1").read().strip().split()
        if len(result) >= 4:
            # macOS df output: filesystem blocks used avail capacity...
            total_blocks = int(result[1])
            used_blocks = int(result[2])
            free_blocks = int(result[3])

            # Convert 512-byte blocks to GB
            total = total_blocks * 512 / 1024 / 1024 / 1024
            used = used_blocks * 512 / 1024 / 1024 / 1024
            free = free_blocks * 512 / 1024 / 1024 / 1024

            return {
                "total": total,
                "used": used,
                "free": free,
                "pct": 100 * used / total if total > 0 else 0,
            }
    except:
        pass
    return {"error": "failed to parse"}


def check_disk_space(label: str = ""):
    """Check disk space and abort if critical."""
    usage = get_disk_usage()
    if "error" in usage:
        print(f"⚠ {label} Could not check disk space")
        return

    free_gb = usage["free"]
    pct = usage["pct"]
    print(f"💾 {label} Disk: {pct:.1f}% used ({free_gb:.1f}GB free)")

    if free_gb < DISK_CRITICAL_THRESHOLD:
        raise RuntimeError(f"DISK CRITICAL: Only {free_gb:.1f}GB free (need {DISK_CRITICAL_THRESHOLD}GB)")


# ── Batch Processing ──────────────────────────────────────────────────────────

def extract_batch_features(batch_idx: int, song_ids: list, pitch_extractor) -> dict:
    """Extract features for one batch of songs.

    Returns dict with keys: feats, roots, qualities, song_ids, metadata
    """
    print(f"\n  Batch {batch_idx}: Extracting features from {len(song_ids)} songs...")

    bb = BillboardDataset(chord_type="majmin")

    batch_feats = []
    batch_roots = []
    batch_qualities = []
    batch_song_ids = []
    batch_metadata = []
    failures = []

    for idx, song_id in enumerate(song_ids):
        try:
            gt = bb.load_track_gt(song_id)
            audio_path = Path(gt["audio_path"])

            if not audio_path.exists():
                failures.append((song_id, "audio_missing"))
                continue

            # Load audio
            y, sr = librosa.load(str(audio_path), sr=22050, mono=True)

            # Extract pitch (cached by PitchExtractor)
            pitch_data = pitch_extractor.extract(str(audio_path))  # (time, 88)

            # Extract chroma
            chroma_raw = librosa.feature.chroma_cqt(
                y=y, sr=sr, n_chroma=12, n_octaves=7, fmin=32.703
            )

            # Extract CQT
            cqt = np.abs(librosa.cqt(y, sr=sr, n_bins=84, fmin=32.703))

            # Beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')

            # Beat-synchronize
            pitch_beat = librosa.util.sync(
                pitch_data.T, librosa.time_to_frames(beats, sr=sr)
            ).T

            chroma_beat = librosa.util.sync(
                chroma_raw, librosa.time_to_frames(beats, sr=sr)
            )

            cqt_beat = librosa.util.sync(
                cqt, librosa.time_to_frames(beats, sr=sr)
            )

            # Extract chord-level features
            for chord in gt["chords"]:
                t0, t1 = chord["t0"], chord["t1"]
                root = chord["root"]
                quality = chord["quality"]
                label = chord["label"]

                beat_mask = (beats >= t0) & (beats < t1)
                if not np.any(beat_mask):
                    continue

                feat88 = np.mean(pitch_beat[:, beat_mask], axis=1)
                feat12 = np.mean(chroma_beat[:, beat_mask], axis=1)
                feat_full = np.concatenate([feat88, feat12])

                batch_feats.append(feat_full)
                batch_roots.append(root)
                batch_qualities.append(quality)
                batch_song_ids.append(song_id)  # Store as string
                batch_metadata.append({"t0": t0, "t1": t1, "label": label})

            if (idx + 1) % 10 == 0:
                print(f"    [{idx+1}/{len(song_ids)}] ✓ {song_id}")

        except Exception as e:
            print(f"    ✗ {song_id}: {e}")
            failures.append((song_id, str(e)))

    print(f"  Batch {batch_idx}: {len(batch_feats)} chord events, {len(failures)} failures")

    return {
        "feats": np.array(batch_feats, dtype=np.float32),
        "roots": np.array(batch_roots, dtype=object),
        "qualities": np.array(batch_qualities, dtype=object),
        "song_ids": np.array(batch_song_ids, dtype=object),  # Keep as strings
        "metadata": batch_metadata,
        "failures": failures,
    }


def process_billboard_batches(force: bool = False):
    """Download/extract/delete Billboard in batches."""
    print(f"\n{'=' * 70}")
    print(f"PHASE 1: Batch-wise Feature Extraction (Disk-Optimized)")
    print(f"{'=' * 70}\n")

    # Check for existing corpus
    if CORPUS_FILE.exists() and not force:
        print(f"✓ Corpus already exists: {CORPUS_FILE}")
        return CORPUS_FILE

    # Initialize dataset
    bb = BillboardDataset(chord_type="majmin")
    bb_mirdata = bb.dataset  # Access underlying mirdata dataset
    all_song_ids = list(bb_mirdata.track_ids)  # Get actual track IDs from mirdata
    print(f"Found {len(all_song_ids)} Billboard tracks")
    np.random.seed(42)
    np.random.shuffle(all_song_ids)

    print(f"Processing {len(all_song_ids)} Billboard songs in batches of {BATCH_SIZE}")
    check_disk_space("Start")

    # Process batches
    pitch_extractor = PitchExtractor()
    all_batch_data = []
    all_failures = []

    for batch_idx in range(0, len(all_song_ids), BATCH_SIZE):
        batch_song_ids = all_song_ids[batch_idx:batch_idx + BATCH_SIZE]
        print(f"\n▶ BATCH {batch_idx // BATCH_SIZE + 1}/{(len(all_song_ids) + BATCH_SIZE - 1) // BATCH_SIZE}")

        try:
            # Extract features
            batch_data = extract_batch_features(batch_idx // BATCH_SIZE, batch_song_ids, pitch_extractor)
            all_batch_data.append(batch_data)
            all_failures.extend(batch_data["failures"])

            # Clean up audio cache
            print(f"  Cleaning up audio files...")
            mir_datasets_path = Path.home() / "mir_datasets" / "billboard"
            if mir_datasets_path.exists():
                # Delete audio files in this batch
                audio_dir = mir_datasets_path / "audio"
                if audio_dir.exists():
                    for song_id in batch_song_ids:
                        # Try to delete audio files for this song
                        # Billboard structure: audio/DECADE/ARTIST/SONG/audio.flac
                        for root, dirs, files in os.walk(audio_dir):
                            for file in files:
                                if file.endswith(('.flac', '.mp3', '.wav', '.m4a')):
                                    try:
                                        os.remove(os.path.join(root, file))
                                    except:
                                        pass
                        break  # Only walk once

            check_disk_space(f"After batch {batch_idx // BATCH_SIZE + 1}")

        except RuntimeError as e:
            if "DISK CRITICAL" in str(e):
                print(f"✗ DISK CRITICAL - aborting")
                raise
            else:
                raise
        except Exception as e:
            print(f"✗ Batch {batch_idx // BATCH_SIZE + 1} error: {e}")
            all_failures.append((f"batch_{batch_idx // BATCH_SIZE}", str(e)))

    # Merge all batches
    print(f"\nMerging {len(all_batch_data)} batches...")
    all_feats = np.vstack([b["feats"] for b in all_batch_data])
    all_roots = np.concatenate([b["roots"] for b in all_batch_data])
    all_qualities = np.concatenate([b["qualities"] for b in all_batch_data])
    all_song_ids_merged = np.concatenate([b["song_ids"] for b in all_batch_data], dtype=object)
    all_metadata_merged = [m for b in all_batch_data for m in b["metadata"]]

    # Remap qualities
    q5_map = {q: i for i, q in enumerate(QUALITY5)}
    q5_map[None] = -1
    quality_idx = np.array([q5_map.get(q, -1) for q in all_qualities], dtype=int)

    # Save corpus
    print(f"\nSaving corpus: {len(all_feats)} chords from {len(set(all_song_ids_merged))} songs")
    np.savez_compressed(
        CORPUS_FILE,
        feats=all_feats,
        roots=all_roots,
        quality_idx=quality_idx,
        song_id=all_song_ids_merged,
        qualities=np.array(QUALITY5, dtype=object),
        metadata=all_metadata_merged,
    )

    print(f"✓ Saved: {CORPUS_FILE}")
    print(f"Failures: {len(all_failures)}")

    check_disk_space("Final")
    return CORPUS_FILE


# ── Training ──────────────────────────────────────────────────────────────────

class QualityHeadMLP(nn.Module):
    def __init__(self, in_dim: int = 100, n_classes: int = 5, h1: int = 128, h2: int = 64):
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

    # Load corpus
    d = np.load(CORPUS_FILE, allow_pickle=True)
    feats = d["feats"].astype(np.float32)
    quality_idx = d["quality_idx"].astype(int)
    song_id = d["song_id"]  # Keep as object (strings)

    # Filter to clean 5-way examples
    keep = quality_idx >= 0
    X = feats[keep]
    y = quality_idx[keep]
    songs = song_id[keep]

    print(f"Clean 5-way training examples: {len(X)}")
    print(f"Class distribution: {np.bincount(y)}")

    # Split by song
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

    print(f"Train: {len(X_train)} samples from {len(train_songs)} songs")
    print(f"Val: {len(X_val)} samples from {len(val_songs)} songs\n")

    # Normalize
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QualityHeadMLP(in_dim=X_train.shape[1], n_classes=5).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Data loaders
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

    # Train
    best_val_acc = 0
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        # Train
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

        # Validate
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
            print(f"Epoch {epoch+1:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save checkpoint
            torch.save(model.state_dict(), QUALITY_MODEL_FILE)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step()

    # Load best model
    model.load_state_dict(torch.load(QUALITY_MODEL_FILE, map_location=device))

    print(f"\n✓ Saved quality model: {QUALITY_MODEL_FILE}")
    print(f"Best val accuracy: {best_val_acc:.3f}")

    return model, X_val_norm, y_val


def train_root_model(epochs: int = 100):
    """Train 12-way root classifier."""
    print(f"\n{'=' * 70}")
    print(f"PHASE 3: Train Root Classifier (12 roots)")
    print(f"{'=' * 70}\n")

    # Load corpus
    d = np.load(CORPUS_FILE, allow_pickle=True)
    feats = d["feats"].astype(np.float32)
    roots = d["roots"]
    song_id = d["song_id"]  # Keep as object (strings)
    quality_idx = d["quality_idx"].astype(int)

    # Filter to chords with quality (roots available)
    keep = quality_idx >= 0
    X = feats[keep]
    y_roots = roots[keep]
    songs = song_id[keep]

    # Convert roots to indices (C=0, C#=1, ..., B=11)
    root_map = {r: i for i, r in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])}
    y = np.array([root_map.get(r, -1) for r in y_roots], dtype=int)
    keep_valid = y >= 0

    X = X[keep_valid]
    y = y[keep_valid]
    songs = songs[keep_valid]

    print(f"Training examples: {len(X)}")
    print(f"Class distribution: {np.bincount(y)}")

    # Split by song
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

    print(f"Train: {len(X_train)} samples from {len(train_songs)} songs")
    print(f"Val: {len(X_val)} samples from {len(val_songs)} songs\n")

    # Normalize
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)

    # Train with scikit-learn
    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    )

    print("Training logistic regression...")
    model.fit(X_train_norm, y_train)

    train_acc = model.score(X_train_norm, y_train)
    val_acc = model.score(X_val_norm, y_val)

    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Val accuracy: {val_acc:.3f}\n")

    # Save model
    np.savez_compressed(
        ROOT_MODEL_FILE.with_suffix('.npz'),
        coef=model.coef_,
        intercept=model.intercept_,
        classes=model.classes_,
    )

    print(f"✓ Saved root model: {ROOT_MODEL_FILE.with_suffix('.npz')}")

    return model, X_val_norm, y_val


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(quality_model, root_model, X_val, y_val_quality, y_val_root):
    """Evaluate on validation set."""
    print(f"\n{'=' * 70}")
    print(f"PHASE 4: Evaluation")
    print(f"{'=' * 70}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    quality_model.eval()
    with torch.no_grad():
        logits = quality_model(torch.tensor(X_val, dtype=torch.float32).to(device))
        quality_pred = logits.argmax(dim=1).cpu().numpy()

    quality_acc = (quality_pred == y_val_quality).mean()
    print(f"Quality accuracy: {quality_acc:.3f}")

    root_acc = root_model.score(X_val, y_val_root)
    print(f"Root accuracy: {root_acc:.3f}")

    return {
        "quality_acc": float(quality_acc),
        "root_acc": float(root_acc),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true", help="Extract features")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate")
    parser.add_argument("--all", action="store_true", help="Do all steps")
    parser.add_argument("--force", action="store_true", help="Force re-extraction")
    args = parser.parse_args()

    if args.all or not (args.extract or args.train or args.evaluate):
        args.extract = args.train = args.evaluate = True

    try:
        # Phase 1: Extract
        if args.extract:
            process_billboard_batches(force=args.force)

        # Phase 2-3: Train
        if args.train:
            quality_model, X_val_quality, y_val_quality = train_quality_model()
            root_model, X_val_root, y_val_root = train_root_model()

        # Phase 4: Evaluate
        if args.evaluate and args.train:
            results = evaluate(quality_model, root_model, X_val_quality, y_val_quality, y_val_root)
            print(f"\nResults: {json.dumps(results, indent=2)}")

    except RuntimeError as e:
        print(f"\n✗ ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
