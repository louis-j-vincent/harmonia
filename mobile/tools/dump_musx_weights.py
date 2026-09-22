"""One-time conversion: the 5 musx torch checkpoints -> 5 flat .npz weight files.

Lets ``mobile/harmonia_mobile/musx_net.py`` (pure numpy) run with NO torch at
inference time. Uses the exact same loader as ``harmonia/musx.py::_nets``
(``NetworkInterface`` over the vendored clone), CPU only (device doesn't
matter for a one-shot weight dump).

Run with the MOBILE venv (needs torch + the clone's mir/ deps):
    ~/harmonia/.venv-mobile/bin/python mobile/tools/dump_musx_weights.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MUSX_DIR = REPO / "third_party" / "musx_ismir2019"
OUT_DIR = Path(__file__).resolve().parents[1] / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ['joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s%d.best' % i
               for i in range(5)]


class InMusxDir:
    def __enter__(self):
        import os
        self._cwd = Path.cwd()
        os.chdir(MUSX_DIR)
        if str(MUSX_DIR) not in sys.path:
            sys.path.insert(0, str(MUSX_DIR))
        return self

    def __exit__(self, *exc):
        import os
        os.chdir(self._cwd)
        return False


def main():
    sys.path.insert(0, str(REPO / "mobile"))
    from harmonia_mobile import musx_net

    with InMusxDir():
        from mir.nn.train import NetworkInterface
        from chordnet_ismir_naive import ChordNet
        for i, name in enumerate(MODEL_NAMES):
            ni = NetworkInterface(ChordNet(None), name, load_checkpoint=False)
            assert ni.finalized, f"fold {i}: weights did not load from .sdict"
            net = ni.net.eval()
            out_path = OUT_DIR / f"musx_fold{i}_weights.npz"
            musx_net.dump_weights_from_torch(net, out_path)
            print(f"fold {i}: wrote {out_path}")


if __name__ == "__main__":
    main()
