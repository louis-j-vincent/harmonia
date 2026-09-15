# Harmonia

Turn a recording into a chord chart. Give it a YouTube link, an uploaded
file, or your microphone; it hears the beat, the chords, and the song's
sections (intro/verse/chorus/bridge…), and writes out a chart you can read,
transpose, annotate, and practise against — served as one small web app,
used from a Mac browser and an iPhone.

```
audio (YouTube / upload / mic)
  → beats (Beat This!)
  → chord posteriors (musx, an ISMIR 2019 ensemble)
  → bars + sections (SongFormer)
  → folding (repeats collapsed into one written block)
  → key + colours
  → chart JSON, served by Flask
```

Personal, single-user project — no accounts, no cloud database, state is
files under `state/`.

## Run it

```bash
source .venv/bin/activate           # or .venv/bin/python directly
python -m harmonia.server           # serves on :7772
```

Open `http://127.0.0.1:7772/` (or the machine's Tailscale address, for the
phone). See `docs/STATE.md` for the full picture: what each module does, the
state layout, the golden report (how to check a change didn't silently move
the library), and where the project's history lives.

## Where things are

- `harmonia/` — the app: `settings.py` (env/config, one place), `pipeline.py`
  (orchestration), the stages (`beats.py`, `musx.py`, `nnls_features.py`,
  `folding.py`, `harmonic_key.py`, `span_rescore.py`), `sections/`
  (SongFormer + shared similarity), `server/` (Flask app + routes).
- `tools/` — operational scripts: `golden.py` (the diff report),
  `avant_apres.py` (the before/after page), `migrate_state.py`, plus
  `tools/sections_bench/` (the section-detection research bench).
- `tests/` — pytest suite.
- `docs/` — `STATE.md` (start here), `known_issues.md` (what's open right
  now), `refactor_2026-09/` (the rewrite's plan and sprint log), `blog/`
  (devlog).
- `third_party/musx_ismir2019/` — vendored chord-recognition model (MIT).
- `archive/` — superseded scripts and one-off migrations, kept for history.

## Develop

```bash
make test      # pytest
make golden    # rebake the library, diff against the frozen baseline
make serve     # python -m harmonia.server
```

## License

MIT
