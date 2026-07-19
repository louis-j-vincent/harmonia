import subprocess, json, sys
from pathlib import Path

TRACKS = [
    ("954", "LaVern Baker", "I Cried A Tear", 156.238367346),
    ("183", "Abba", "Chiquitita", 324.675918367),
    ("44", "The Power Station", "Some Like It Hot", 305.841632653),
    ("1111", "Chris Kenner", "Land Of 1000 Dances", 149.133061224),
    ("406", "Pure Prairie League", "Amie", 260.59755102),
    ("362", "Wednesday", "Last Kiss", 156.081632653),
    ("334", "Rockwell", "Somebody's Watching Me", 241.65877551),
    ("217", "Rick Springfield", "Jessie's Girl", 196.503287981),
    ("1104", "Tina Turner", "What You Get Is What You See", 267.232675736),
    ("168", "The Animals", "San Franciscan Nights", 200.646530612),
]

YTDLP = ".venv/bin/yt-dlp"

def search(query, n=5):
    cmd = [YTDLP, f"ytsearch{n}:{query}", "--print", "%(id)s\t%(duration)s\t%(title)s",
           "--skip-download", "--no-warnings"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    lines = [l for l in r.stdout.strip().split("\n") if l.strip()]
    out = []
    for l in lines:
        parts = l.split("\t")
        if len(parts) != 3:
            continue
        vid, dur, title = parts
        try:
            dur = float(dur)
        except ValueError:
            dur = None
        out.append((vid, dur, title))
    return out, r.stderr

results = {}
for tid, artist, title, gt_dur in TRACKS:
    query = f"{artist} {title}"
    print(f"=== {tid} {query} (GT dur={gt_dur:.1f}s) ===", flush=True)
    try:
        cands, err = search(query)
    except Exception as e:
        print("  SEARCH FAILED:", e)
        results[tid] = {"error": str(e)}
        continue
    best = None
    for vid, dur, ctitle in cands:
        if dur is None:
            continue
        diff = abs(dur - gt_dur)
        tol = max(0.05 * gt_dur, 5.0)
        ok = diff <= tol
        marker = "OK" if ok else "  "
        print(f"  [{marker}] {vid}  dur={dur:.1f}  diff={diff:.1f}  tol={tol:.1f}  {ctitle}")
        if ok and (best is None or diff < best[1]):
            best = (vid, diff, dur, ctitle)
    results[tid] = {"artist": artist, "title": title, "gt_dur": gt_dur,
                     "best": best, "candidates": cands}

with open(Path(__file__).parent / "billboard_search_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n\n=== SUMMARY ===")
hits = 0
for tid, r in results.items():
    if r.get("best"):
        hits += 1
        print(f"{tid}: HIT  {r['best'][0]}  diff={r['best'][1]:.1f}s")
    else:
        print(f"{tid}: MISS")
print(f"\nHit rate: {hits}/{len(TRACKS)}")
