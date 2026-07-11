"""One-off migration #3: fold the rotor's sound+pulse calls into a shared
feedback() helper that also fires navigator.vibrate() (no-op on iOS Safari,
works on Android PWAs). Straight string swap on already-rendered chart HTML,
no re-inference.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO / "docs" / "plots"

OLD_HELPER = """  function pulsePointer(){
    pointer.classList.remove("tick");
    void pointer.offsetWidth;  // restart the CSS animation
    pointer.classList.add("tick");
  }"""

NEW_HELPER = """  function pulsePointer(){
    pointer.classList.remove("tick");
    void pointer.offsetWidth;  // restart the CSS animation
    pointer.classList.add("tick");
  }
  function feedback(){
    // sound + visual pulse everywhere; vibration is a no-op on iOS Safari
    // (no Vibration API there) but fires on Android PWAs/Chrome.
    playTick(); pulsePointer();
    if(navigator.vibrate) navigator.vibrate(6);
  }"""

OLD_NOTCH = "if(notch!==lastNotch){ lastNotch=notch; playTick(); pulsePointer(); }"
NEW_NOTCH = "if(notch!==lastNotch){ lastNotch=notch; feedback(); }"

OLD_RELEASE = """    applyRot(rot,true);
    playTick(); pulsePointer();
    field.value=off;"""
NEW_RELEASE = """    applyRot(rot,true);
    feedback();
    field.value=off;"""


def main() -> None:
    files = sorted(PLOTS_DIR.glob("inferred_*.html"))
    patched, skipped = 0, 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        if OLD_HELPER not in text or OLD_NOTCH not in text or OLD_RELEASE not in text:
            skipped += 1
            continue
        text = (text.replace(OLD_HELPER, NEW_HELPER, 1)
                     .replace(OLD_NOTCH, NEW_NOTCH, 1)
                     .replace(OLD_RELEASE, NEW_RELEASE, 1))
        f.write_text(text, encoding="utf-8")
        patched += 1
        print(f"patched {f.name}")
    print(f"\n{patched} patched, {skipped} already up to date / not matching")


if __name__ == "__main__":
    sys.exit(main())
