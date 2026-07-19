"""Laplacian structural segmentation (McFee & Ellis 2014) via librosa — the
canonical online recurrence-matrix / diagonal-stripe repetition detector.
Reports boundaries (seconds) + repeat labels (same int = same recurring section).
"""
import sys
import numpy as np
import librosa
import scipy.cluster.hierarchy
from sklearn.cluster import KMeans

def analyze(path, n_types=6):
    y, sr = librosa.load(path, sr=22050, mono=True)
    dur = len(y) / sr
    # beat-synchronous CQT chroma
    C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=7*12*3))
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    Csync = librosa.util.sync(C, beats, aggregate=np.median)
    # weighted recurrence matrix (affinity), width suppresses main diagonal
    R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity', sym=True)
    # enhance diagonals (repeated sections show as off-diagonal stripes)
    from scipy.ndimage import median_filter
    Rf = median_filter(R, size=(1, 7))
    # sequence (path) matrix: link consecutive beats
    path_dist = np.sum(np.diff(Csync, axis=1)**2, axis=0)
    sigma = np.median(path_dist)
    path_sim = np.exp(-path_dist / (sigma + 1e-9))
    R_path = np.diag(path_sim, 1) + np.diag(path_sim, -1)
    # balanced combination
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)
    mu = deg_path.dot(deg_path + deg_rec) / (np.sum((deg_path + deg_rec)**2) + 1e-9)
    A = mu * Rf + (1 - mu) * R_path
    # symmetric normalized Laplacian -> eigvecs
    Dinv = np.diag(1.0 / (np.sum(A, axis=1) + 1e-9)**0.5)
    L = np.eye(A.shape[0]) - Dinv.dot(A).dot(Dinv)
    evals, evecs = scipy.linalg.eigh(L) if False else np.linalg.eigh(L)
    # use first n_types eigenvectors, row-normalize, k-means -> frame labels
    X = evecs[:, :n_types]
    Xn = librosa.util.normalize(X, norm=2, axis=1)
    km = KMeans(n_clusters=n_types, n_init=10, random_state=0)
    seg_ids = km.fit_predict(Xn)
    # temporal smoothing: majority vote over a ~16-beat window kills the flicker
    from scipy.ndimage import median_filter
    seg_ids = median_filter(seg_ids, size=17, mode='nearest')
    # convert beat-frame labels to time boundaries
    bt = librosa.frames_to_time(beats, sr=sr)
    # segment where label changes
    bounds_idx = [0] + list(np.flatnonzero(np.diff(seg_ids)) + 1)
    segs = []
    for i, b in enumerate(bounds_idx):
        start_beat = b
        end_beat = bounds_idx[i+1] if i+1 < len(bounds_idx) else len(seg_ids)
        t0 = bt[start_beat] if start_beat < len(bt) else dur
        segs.append((round(float(t0), 1), int(seg_ids[b])))
    return dur, float(np.atleast_1d(tempo)[0]), segs, R

if __name__ == "__main__":
    import scipy.linalg
    path = sys.argv[1]
    ntypes = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    dur, tempo, segs, R = analyze(path, ntypes)
    print(f"file={path.split('/')[-1]} dur={dur:.1f}s tempo={tempo:.0f}bpm nseg={len(segs)}")
    from collections import Counter
    print("label counts:", dict(Counter(l for _, l in segs)))
    print("segments (t0_s, label):")
    for t0, l in segs:
        print(f"  {t0:7.1f}  {l}")
    # off-diagonal repetition energy = evidence of repeats
    off = R.copy(); np.fill_diagonal(off, 0)
    print(f"recurrence off-diagonal density={off.mean():.4f} (repeat stripes present if >0)")
