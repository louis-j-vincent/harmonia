"""
Phase B: Context-Window Quality Decoder
Tests 3 architectures × 3 normalizations for chord quality prediction using learned context.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG & DATA STRUCTURES
# ============================================================================

DATA_DIR = Path('/Users/vincente/Documents/Projets Perso/Code/harmonia/data/cache')
OUTPUT_DIR = Path('/Users/vincente/Documents/Projets Perso/Code/harmonia/data/models')
PLOTS_DIR = Path('/Users/vincente/Documents/Projets Perso/Code/harmonia/docs/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT_SIZE = 3  # 3 before + current + 3 after = 7 total
CHROMA_DIM = 12
NUM_CLASSES = 5
CLASS_NAMES = ['maj', 'min', 'dom', 'hdim', 'dim']
CLASS_WEIGHTS = {'maj': 1.0, 'min': 1.0, 'dom': 2.5, 'hdim': 2.0, 'dim': 0.5}

@dataclass
class TrainResult:
    architecture: str
    normalization: str
    balanced_acc_train: float
    balanced_acc_val: float
    balanced_acc_test: float
    recalls_test: Dict[str, float]  # per-class
    model_path: str

    def to_dict(self):
        return {
            'architecture': self.architecture,
            'normalization': self.normalization,
            'balanced_acc_train': float(self.balanced_acc_train),
            'balanced_acc_val': float(self.balanced_acc_val),
            'balanced_acc_test': float(self.balanced_acc_test),
            'recalls_test': {k: float(v) for k, v in self.recalls_test.items()},
            'model_path': str(self.model_path),
        }

# ============================================================================
# DATA LOADING & ALIGNMENT
# ============================================================================

def load_data():
    """Load Billboard and bass prediction data with proper alignment."""
    print("Loading Billboard corpus...")
    billboard = np.load(DATA_DIR / 'billboard/billboard_training_corpus_full.npz', allow_pickle=True)

    print("Loading bass predictions...")
    bass_pred = np.load(DATA_DIR / 'bass_predictions_train_val_test.npz', allow_pickle=True)

    return billboard, bass_pred

def align_data(billboard, bass_pred):
    """
    Align Billboard data with bass predictions.

    Bass predictions are indexed by song_id; we need to match with Billboard's song_id.
    Since chord counts may differ between datasets, we use positional indexing within songs.
    """

    # Build song-wise index for Billboard
    billboard_by_song = {}
    for idx, song_id in enumerate(billboard['song_id']):
        if song_id not in billboard_by_song:
            billboard_by_song[song_id] = []
        billboard_by_song[song_id].append(idx)

    print(f"Billboard: {len(billboard_by_song)} songs, {len(billboard['song_id'])} total chords")

    # Process each split
    result = {}
    for split_name, pred_key in [('train', 'chroma_only_train'),
                                  ('val', 'chroma_only_val'),
                                  ('test', 'chroma_only_test')]:

        song_ids = bass_pred[f'{pred_key}_song']
        root_probs = bass_pred[f'{pred_key}_prob']

        # For each song, find matching chords in Billboard
        valid_indices_bb = []  # Billboard indices
        valid_indices_bass = []  # Bass prediction indices

        # Build per-song index for bass predictions
        bass_by_song = {}
        for idx, song_id in enumerate(song_ids):
            if song_id not in bass_by_song:
                bass_by_song[song_id] = []
            bass_by_song[song_id].append(idx)

        # Match chords: use min of both counts to avoid index out of bounds
        for song_id in sorted(set(song_ids)):
            if song_id not in billboard_by_song:
                continue  # Song not in Billboard

            bb_indices = billboard_by_song[song_id]
            bass_indices = bass_by_song[song_id]

            # Use minimum count to avoid misalignment
            n_common = min(len(bb_indices), len(bass_indices))
            valid_indices_bb.extend(bb_indices[:n_common])
            valid_indices_bass.extend(bass_indices[:n_common])

        valid_indices_bb = np.array(valid_indices_bb)
        valid_indices_bass = np.array(valid_indices_bass)

        print(f"{split_name}: {len(valid_indices_bb)} aligned chords from {len(set(song_ids))} songs")

        # Extract features and labels
        result[split_name] = {
            'chroma': billboard['feats'][valid_indices_bb],
            'y': billboard['quality_idx'][valid_indices_bb],
            'song_ids': billboard['song_id'][valid_indices_bb],
            'root_prob': root_probs[valid_indices_bass],
        }

    return result

# ============================================================================
# CONTEXT WINDOW EXTRACTION & NORMALIZATION
# ============================================================================

def extract_context_windows(split_data: Dict) -> Dict:
    """
    Extract context windows for each chord.
    For each chord at position i, extract:
    - Neighbors [i-3, i-2, i-1, i, i+1, i+2, i+3]
    - Each neighbor's root probability distribution (12-dim)

    Returns dict with song_id mapped lists of indices for efficient extraction.
    """
    result = {}
    for split_name, split in split_data.items():
        song_ids = split['song_ids']
        root_probs = split['root_prob']

        # Group by song for context extraction
        song_groups = {}
        for idx, song_id in enumerate(song_ids):
            if song_id not in song_groups:
                song_groups[song_id] = []
            song_groups[song_id].append(idx)

        # Extract contexts per song
        contexts = []  # list of 7x12 matrices
        valid_indices = []  # indices that have full context

        for song_id, indices in sorted(song_groups.items()):
            for i, idx in enumerate(indices):
                # Check if we have full context (3 before and 3 after)
                if i < CONTEXT_SIZE or i >= len(indices) - CONTEXT_SIZE:
                    continue

                # Get context: [i-3, i-2, i-1, i, i+1, i+2, i+3]
                context_idx_local = list(range(i - CONTEXT_SIZE, i + CONTEXT_SIZE + 1))
                context_idx_global = [indices[j] for j in context_idx_local]

                # Stack root probabilities (7x12)
                context = root_probs[context_idx_global]  # 7x12
                contexts.append(context)
                valid_indices.append(idx)

        result[split_name] = {
            'contexts': np.array(contexts),  # (N, 7, 12)
            'indices': np.array(valid_indices),
        }

    return result

def apply_normalization(contexts, normalization_type='raw'):
    """
    Apply context normalization.

    Args:
        contexts: (N, 7, 12) array of root probability distributions
        normalization_type: 'raw' | 'relative-key' | 'relative-root'

    Returns:
        (N, 7, 12) normalized contexts
    """
    if normalization_type == 'raw':
        return contexts

    elif normalization_type == 'relative-root':
        # Shift all neighbors relative to current (center) root
        # Current root is at position 3 (center)
        # Current root is the highest probability root
        current_roots = np.argmax(contexts[:, 3, :], axis=1)  # (N,)

        normalized = contexts.copy()
        for n in range(contexts.shape[0]):
            curr_root = current_roots[n]
            # Shift all distributions by -curr_root
            for t in range(7):
                normalized[n, t, :] = np.roll(contexts[n, t, :], -curr_root)

        return normalized

    elif normalization_type == 'relative-key':
        # For now, estimate key from current chord's root distribution
        # This is a simplification: we'll use current root as proxy for key
        # A proper implementation would estimate key from chord sequence
        current_roots = np.argmax(contexts[:, 3, :], axis=1)  # (N,)

        normalized = contexts.copy()
        for n in range(contexts.shape[0]):
            key = current_roots[n]  # Proxy for key
            # Shift all distributions by -key
            for t in range(7):
                normalized[n, t, :] = np.roll(contexts[n, t, :], -key)

        return normalized

    else:
        raise ValueError(f"Unknown normalization: {normalization_type}")

# ============================================================================
# BASELINE: CHROMA-ONLY (NO CONTEXT)
# ============================================================================

def train_baseline_model(split_data):
    """Train baseline model using only current chord chroma (no context)."""
    print("\n" + "="*70)
    print("BASELINE: Chroma-only model (no context)")
    print("="*70)

    # Get chroma for valid indices only (same as context extraction)
    context_windows = extract_context_windows(split_data)

    X_train = split_data['train']['chroma'][context_windows['train']['indices']]
    X_val = split_data['val']['chroma'][context_windows['val']['indices']]
    X_test = split_data['test']['chroma'][context_windows['test']['indices']]

    y_train = split_data['train']['y'][context_windows['train']['indices']]
    y_val = split_data['val']['y'][context_windows['val']['indices']]
    y_test = split_data['test']['y'][context_windows['test']['indices']]

    # Build simple model
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(12,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=CLASS_WEIGHTS,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )

    # Evaluate
    y_pred_train = np.argmax(model.predict(X_train, verbose=0), axis=1)
    y_pred_val = np.argmax(model.predict(X_val, verbose=0), axis=1)
    y_pred_test = np.argmax(model.predict(X_test, verbose=0), axis=1)

    acc_train = balanced_accuracy_score(y_train, y_pred_train)
    acc_val = balanced_accuracy_score(y_val, y_pred_val)
    acc_test = balanced_accuracy_score(y_test, y_pred_test)

    recalls_test = {}
    per_class_recalls = recall_score(y_test, y_pred_test, average=None, labels=range(NUM_CLASSES), zero_division=0)
    for i, class_name in enumerate(CLASS_NAMES):
        recalls_test[class_name] = per_class_recalls[i]

    print(f"Train balanced acc: {acc_train:.4f}")
    print(f"Val balanced acc: {acc_val:.4f}")
    print(f"Test balanced acc: {acc_test:.4f}")
    print(f"Test per-class recalls: {recalls_test}")

    return TrainResult(
        architecture='baseline',
        normalization='chroma-only',
        balanced_acc_train=acc_train,
        balanced_acc_val=acc_val,
        balanced_acc_test=acc_test,
        recalls_test=recalls_test,
        model_path='baseline'
    )

# ============================================================================
# ARCHITECTURE 1: CNN
# ============================================================================

def build_cnn_model() -> models.Model:
    """Build CNN over context window."""
    model = models.Sequential([
        # Input: (7, 12)
        layers.Conv1D(16, kernel_size=3, activation='relu', input_shape=(7, 12)),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(32, kernel_size=2, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# ============================================================================
# ARCHITECTURE 2: LSTM
# ============================================================================

def build_lstm_model() -> models.Model:
    """Build LSTM over context window."""
    model = models.Sequential([
        # Input: (7, 12)
        layers.LSTM(32, input_shape=(7, 12), return_sequences=False),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# ============================================================================
# ARCHITECTURE 3: TRANSFORMER (Multihead Attention)
# ============================================================================

def build_transformer_model() -> models.Model:
    """Build Transformer with multihead attention."""
    inputs = layers.Input(shape=(7, 12))

    # Position encoding
    pos_encoding = keras.initializers.TruncatedNormal(stddev=0.1)(
        tf.range(7, dtype=tf.float32)[None, :, None]
    )
    x = inputs + pos_encoding[:, :, :12]

    # Multihead attention
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    # Feed-forward
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(12)(x)
    x = layers.Add()([x, attn])  # residual
    x = layers.LayerNormalization()(x)

    # Global pooling and classification
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_architecture(
    split_data: Dict,
    contexts_dict: Dict,
    arch_name: str,
    normalization: str,
    results: List[TrainResult]
):
    """Train a single architecture with a given normalization."""
    print(f"\n{'='*70}")
    print(f"Architecture: {arch_name.upper()} | Normalization: {normalization}")
    print(f"{'='*70}")

    # Build dataset
    X_train_ctx = apply_normalization(contexts_dict['train']['contexts'], normalization)
    X_val_ctx = apply_normalization(contexts_dict['val']['contexts'], normalization)
    X_test_ctx = apply_normalization(contexts_dict['test']['contexts'], normalization)

    y_train = split_data['train']['y'][contexts_dict['train']['indices']]
    y_val = split_data['val']['y'][contexts_dict['val']['indices']]
    y_test = split_data['test']['y'][contexts_dict['test']['indices']]

    # Build model
    if arch_name == 'cnn':
        model = build_cnn_model()
    elif arch_name == 'lstm':
        model = build_lstm_model()
    elif arch_name == 'transformer':
        model = build_transformer_model()
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # Train
    history = model.fit(
        X_train_ctx, y_train,
        validation_data=(X_val_ctx, y_val),
        epochs=50,
        batch_size=32,
        class_weight=CLASS_WEIGHTS,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )

    # Evaluate
    y_pred_train = np.argmax(model.predict(X_train_ctx, verbose=0), axis=1)
    y_pred_val = np.argmax(model.predict(X_val_ctx, verbose=0), axis=1)
    y_pred_test = np.argmax(model.predict(X_test_ctx, verbose=0), axis=1)

    acc_train = balanced_accuracy_score(y_train, y_pred_train)
    acc_val = balanced_accuracy_score(y_val, y_pred_val)
    acc_test = balanced_accuracy_score(y_test, y_pred_test)

    recalls_test = {}
    per_class_recalls = recall_score(y_test, y_pred_test, average=None, labels=range(NUM_CLASSES), zero_division=0)
    for i, class_name in enumerate(CLASS_NAMES):
        recalls_test[class_name] = per_class_recalls[i]

    print(f"Train balanced acc: {acc_train:.4f}")
    print(f"Val balanced acc: {acc_val:.4f}")
    print(f"Test balanced acc: {acc_test:.4f}")
    print(f"Test per-class recalls: {recalls_test}")

    # Save model
    model_path = OUTPUT_DIR / f'{arch_name}_{normalization.replace("-", "_")}.h5'
    model.save(model_path)

    result = TrainResult(
        architecture=arch_name,
        normalization=normalization,
        balanced_acc_train=acc_train,
        balanced_acc_val=acc_val,
        balanced_acc_test=acc_test,
        recalls_test=recalls_test,
        model_path=str(model_path)
    )
    results.append(result)

    return result

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_results_plot(results: List[TrainResult]):
    """Create ablation heatmap and confusion matrices."""

    # Organize results
    architectures = sorted(set(r.architecture for r in results if r.architecture != 'baseline'))
    normalizations = sorted(set(r.normalization for r in results if r.normalization != 'chroma-only'))
    baseline = next((r for r in results if r.architecture == 'baseline'), None)

    # Create heatmaps: balanced_acc and dom_recall
    balanced_acc_data = np.zeros((len(architectures), len(normalizations)))
    dom_recall_data = np.zeros((len(architectures), len(normalizations)))

    for i, arch in enumerate(architectures):
        for j, norm in enumerate(normalizations):
            result = next((r for r in results if r.architecture == arch and r.normalization == norm), None)
            if result:
                balanced_acc_data[i, j] = result.balanced_acc_test
                dom_recall_data[i, j] = result.recalls_test.get('dom', 0)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Balanced accuracy heatmap
    sns.heatmap(balanced_acc_data, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=normalizations, yticklabels=architectures, ax=axes[0],
                vmin=0.7, vmax=0.8, cbar_kws={'label': 'Balanced Accuracy'})
    axes[0].set_title('Balanced Accuracy (Test Set)')
    if baseline:
        axes[0].text(0.5, -0.15, f'Baseline (chroma-only): {baseline.balanced_acc_test:.4f}',
                    ha='center', transform=axes[0].transAxes)

    # Dom recall heatmap
    sns.heatmap(dom_recall_data, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=normalizations, yticklabels=architectures, ax=axes[1],
                vmin=0.6, vmax=0.75, cbar_kws={'label': 'Dom Recall'})
    axes[1].set_title('Dominant Recall (Test Set)')
    if baseline:
        axes[1].text(0.5, -0.15, f'Baseline (chroma-only): {baseline.recalls_test.get("dom", 0):.4f}',
                    ha='center', transform=axes[1].transAxes)

    plt.tight_layout()
    plot_path = PLOTS_DIR / 'phase_b_quality_context_ablation.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved ablation plot: {plot_path}")

    # Convert to HTML
    html_path = PLOTS_DIR / 'phase_b_quality_context_ablation.html'
    create_html_results(results, html_path, balanced_acc_data, dom_recall_data,
                        architectures, normalizations)

def create_html_results(results, html_path, balanced_acc_data, dom_recall_data,
                        architectures, normalizations):
    """Create interactive HTML results table."""

    baseline = next((r for r in results if r.architecture == 'baseline'), None)

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Phase B Results: Context Decoder</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
            h1 { color: #333; }
            .baseline-box { background: #e3f2fd; padding: 15px; border-left: 4px solid #2196F3; margin: 20px 0; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: right; }
            th { background: #f0f0f0; font-weight: bold; }
            td:first-child, th:first-child { text-align: left; }
            tr:nth-child(even) { background: #f9f9f9; }
            .best { background: #c8e6c9; font-weight: bold; }
            .metric-name { font-weight: 500; color: #555; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Phase B: Context-Window Quality Decoder</h1>
            <p>Evaluating learned architectures on chord quality prediction with root context.</p>
    """

    if baseline:
        html += f"""
            <div class="baseline-box">
                <h2>Baseline (Chroma-only, No Context)</h2>
                <p><strong>Balanced Accuracy:</strong> {baseline.balanced_acc_test:.4f}</p>
                <p><strong>Dominant Recall:</strong> {baseline.recalls_test.get('dom', 0):.4f}</p>
                <p>Per-class recalls: maj={baseline.recalls_test.get('maj', 0):.4f},
                   min={baseline.recalls_test.get('min', 0):.4f},
                   dom={baseline.recalls_test.get('dom', 0):.4f},
                   hdim={baseline.recalls_test.get('hdim', 0):.4f},
                   dim={baseline.recalls_test.get('dim', 0):.4f}</p>
            </div>
        """

    html += """
            <h2>Results Summary</h2>
            <table>
                <tr>
                    <th>Architecture</th>
                    <th>Normalization</th>
                    <th>Train Balanced Acc</th>
                    <th>Val Balanced Acc</th>
                    <th>Test Balanced Acc</th>
                    <th>Dom Recall</th>
                    <th>Maj Recall</th>
                    <th>Min Recall</th>
                    <th>Hdim Recall</th>
                    <th>Dim Recall</th>
                </tr>
    """

    for result in sorted(results, key=lambda r: r.balanced_acc_test, reverse=True):
        if result.architecture == 'baseline':
            continue
        html += f"""
                <tr>
                    <td>{result.architecture}</td>
                    <td>{result.normalization}</td>
                    <td>{result.balanced_acc_train:.4f}</td>
                    <td>{result.balanced_acc_val:.4f}</td>
                    <td class="best">{result.balanced_acc_test:.4f}</td>
                    <td>{result.recalls_test.get('dom', 0):.4f}</td>
                    <td>{result.recalls_test.get('maj', 0):.4f}</td>
                    <td>{result.recalls_test.get('min', 0):.4f}</td>
                    <td>{result.recalls_test.get('hdim', 0):.4f}</td>
                    <td>{result.recalls_test.get('dim', 0):.4f}</td>
                </tr>
        """

    html += """
            </table>
        </div>
    </body>
    </html>
    """

    with open(html_path, 'w') as f:
        f.write(html)
    print(f"Saved HTML results: {html_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("PHASE B: Context-Window Quality Decoder")
    print("="*70)

    # Load data
    billboard, bass_pred = load_data()
    split_data = align_data(billboard, bass_pred)

    # Extract context windows
    print("\nExtracting context windows...")
    contexts_dict = extract_context_windows(split_data)
    print(f"Train contexts: {contexts_dict['train']['contexts'].shape}")
    print(f"Val contexts: {contexts_dict['val']['contexts'].shape}")
    print(f"Test contexts: {contexts_dict['test']['contexts'].shape}")

    # Train baseline
    results = []
    baseline_result = train_baseline_model(split_data)
    results.append(baseline_result)

    # Train architectures
    architectures = ['cnn', 'lstm', 'transformer']
    normalizations = ['raw', 'relative-key', 'relative-root']

    for arch in architectures:
        for norm in normalizations:
            try:
                result = train_architecture(split_data, contexts_dict, arch, norm, results)
            except Exception as e:
                print(f"ERROR training {arch} with {norm}: {e}")
                import traceback
                traceback.print_exc()

    # Save results
    results_json = [r.to_dict() for r in results]
    results_path = OUTPUT_DIR / 'phase_b_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved results: {results_path}")

    # Create visualizations
    create_results_plot(results)

    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print("\nBaseline (chroma-only):")
    baseline = next((r for r in results if r.architecture == 'baseline'), None)
    if baseline:
        print(f"  Balanced Acc: {baseline.balanced_acc_test:.4f}")
        print(f"  Dom Recall: {baseline.recalls_test.get('dom', 0):.4f}")

    print("\nBest context model:")
    best = max((r for r in results if r.architecture != 'baseline'),
               key=lambda r: r.recalls_test.get('dom', 0))
    print(f"  {best.architecture.upper()} + {best.normalization}")
    print(f"  Balanced Acc: {best.balanced_acc_test:.4f}")
    print(f"  Dom Recall: {best.recalls_test.get('dom', 0):.4f}")

    if baseline:
        improvement = best.recalls_test.get('dom', 0) - baseline.recalls_test.get('dom', 0)
        print(f"\n  Improvement over baseline: {improvement:+.4f}")

if __name__ == '__main__':
    main()
