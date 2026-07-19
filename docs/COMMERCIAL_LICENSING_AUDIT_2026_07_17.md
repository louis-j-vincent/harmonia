# Commercial Licensing Audit — Harmonia

**Date:** 2026-07-17
**Scope:** Third-party models, libraries, and data sources in the production
pipeline, assessed for a **paid/commercial** product launch.
**Method:** Every license claim below was verified by reading the actual
`LICENSE`/`NOTICE`/`METADATA` file in the installed `.venv`, the bundled repo,
or the upstream source — not from memory. Sources are cited per row.

> **⚠️ This is preliminary engineering research, NOT legal advice.** I am not a
> lawyer. Several items below (GPL linkage in SaaS, model-weights vs.
> training-data provenance, non-commercial dataset terms) are genuinely
> unsettled legal questions where the answer depends on facts about how you
> ship and distribute. **Before charging money for this product, have an actual
> IP / entertainment-technology lawyer review this document and the two
> flagged blockers.** Do not treat the green "OK" rows as a substitute for that
> review.

---

## 1. Executive summary — the bottom line

**Clearly fine (permissive, commercial-OK, ship as-is):**
Basic Pitch (Apache 2.0), librosa (ISC), music-x-lab ISMIR2019 wrapper code
(MIT), yt-dlp *the software* (Unlicense/public domain), the `vamp` Python
wrapper (MIT), torchaudio (BSD-2), music21 (BSD-3), mir_eval (MIT). These carry
only attribution/notice obligations at most.

**Two real concerns that need action before launch:**

1. **madmom — NON-COMMERCIAL license (`CC BY-NC-SA 4.0`).** This is the
   single most clear-cut blocker. The PyPI page lists "BSD" but the **actual
   `LICENSE` file states every file in the repo is CC BY-NC-SA 4.0** and
   explicitly says: *"If you want to include any of these files … in a
   commercial product, please contact Gerhard Widmer."* madmom is used in
   production (`harmonia/models/rhythm.py`, beat/downbeat tracking). **You
   cannot ship madmom in a paid product without a separate commercial license
   from JKU/Widmer, or replacing it** (the librosa beat tracker is already a
   `--no-madmom` fallback path).

2. **NNLS-Chroma / Chordino VAMP plugin — GPL v2-or-later (copyleft).** This is
   the default production feature front-end (`HARMONIA_ANALYZE_FRONTEND=nnls24`,
   via `harmonia/models/nnls_features.py`). GPL's obligations trigger on
   **distribution**. If Harmonia is a **server-side SaaS** and you never ship
   the plugin binary to users, GPL v2 is likely *not* triggered (it is not
   AGPL — mere "use" over a network isn't "conveying"). **But** the moment you
   distribute a desktop/bundled build containing the plugin, or link it into
   distributed code, you inherit GPL copyleft on the combined work. This needs
   a lawyer's call on your specific deployment shape.

**Needs verification but probably manageable:**
- **music-x-lab pretrained weights** (bundled, MIT-licensed repo) were trained
  on the Humphrey & Bello chord corpus (annotations over Billboard/Isophonics
  et al., copyrighted commercial audio collected privately). The code+weights
  wrapper is MIT; the *training-data provenance* is a theoretical, legally
  untested concern (see §4).
- **Your own in-house weights** (`yt_chord_model*.npz`, `nnls24_heads.npz`,
  `ctx_v3.npz`) were trained on POP909 and YouTube-sourced audio — POP909 is
  research-only; see §4.

**Not in the production path (flagged for completeness):** Demucs / torchaudio
HDemucs — code is MIT/BSD but the **pretrained separation weights are trained
on MUSDB18-HQ, which is non-commercial/research-only**. Currently only used in
`scripts/` experiments, not the server. Do not promote it to production without
resolving the weights license.

**YouTube audio sourcing** is a copyright/ToS question, not a software-license
one — covered briefly in §3 per your design decision (mirror Chord AI's
user-supplied-URL model).

---

## 2. Per-dependency table

Legend: **Commercial?** — can it ship in a paid product as-is.
"Code vs. weights" flags where the license on shipped model weights differs (or
may differ) from the code license.

| # | Dependency (version) | License (verified) | Commercial? | Key restrictions | Code vs. weights note | In production? |
|---|---|---|---|---|---|---|
| 1 | **Basic Pitch** 0.3.0 (Spotify) | **Apache 2.0** | ✅ Yes | Attribution/NOTICE retention; Apache patent grant (favorable) | Weights (`ICASSP_2022_MODEL_PATH`) ship inside the pip package under the **same** Apache-2.0 license. NOTICE lists bundled ISC/MIT/BSD deps. | ✅ core pitch extraction |
| 2 | **librosa** 0.11.0 | **ISC** | ✅ Yes | Attribution only | N/A (no weights) | ✅ audio analysis |
| 3 | **madmom** 0.16.1 | **CC BY-NC-SA 4.0** (NonCommercial) | ❌ **NO** — blocker | *NonCommercial*, ShareAlike, Attribution. Must contact G. Widmer/JKU for commercial license. PyPI "BSD" classifier is misleading — LICENSE file governs. | Ships its own trained beat/downbeat models — all under the same NC license. | ✅ `rhythm.py` beat/downbeat (has librosa fallback) |
| 4 | **NNLS-Chroma / Chordino** VAMP plugin (c4dm/QMUL) | **GPL v2-or-later** | ⚠️ Conditional | Copyleft. Triggers on **distribution** of the binary/linked work. SaaS-only use likely OK (GPL≠AGPL); bundling/desktop distribution triggers GPL on the combined work. | Native plugin invoked out-of-library via the MIT `vamp` wrapper. No separate weights. | ✅ default front-end (`nnls_features.py`, `HARMONIA_ANALYZE_FRONTEND=nnls24`) |
| 5 | **`vamp`** Python wrapper (vampyhost) 1.1.0 | **MIT** | ✅ Yes | Attribution | The wrapper is MIT, but at runtime it **loads the GPL NNLS-Chroma plugin** (see row 4 for the real constraint). | ✅ hosts the plugin |
| 6 | **music-x-lab ISMIR2019-Large-Vocab-Chord-Recognition** (bundled `third_party/`) | **MIT** (Copyright 2023 Music X Lab) | ✅ Yes (code) | Attribution | **Verified: LICENSE is MIT and covers the repo, incl. bundled weights (`data/cross_weight*.pkl`).** Weights trained on the *Humphrey & Bello* corpus (Billboard/Isophonics-derived; copyrighted audio collected privately). Training-data provenance is a theoretical, untested concern — see §4. | ✅ root/quality + bass (`musx_bass.py`, DEPLOY-3 default) |
| 7 | **BTC-ISMIR19** (jayg996) | **MIT** | ✅ Yes (code) | Attribution | Trained on Isophonics / Robbie Williams / UsPop2002 (annotations only; audio *not* shipped, "collected from online providers"). Same training-data caveat as row 6. | ❌ tested, not deployed |
| 8 | **yt-dlp** 2026.7.4 | **Unlicense** (public domain) | ✅ Yes | None on the software itself | N/A. **Separate issue:** what you *download* with it — see §3. | ✅ ingest (user-supplied URLs) |
| 9 | **Demucs** 4.1.0 (Meta) | **MIT** (code) | ⚠️ code yes / weights no | — | **Pretrained weights (`htdemucs`, `hdemucs_mmi`) trained on MUSDB18-HQ, which is non-commercial/research-only.** Code MIT ≠ weights license. | ❌ `scripts/` experiments only |
| 10 | **torchaudio** 2.11.0 (HDemucs bundle) | **BSD-2-Clause** | ⚠️ code yes / weights no | Attribution | `HDEMUCS_HIGH_MUSDB_PLUS` bundle weights are also MUSDB18-derived → same non-commercial caveat as row 9. | ❌ experiments only |
| 11 | **music21** 10.5.0 | **BSD-3-Clause** | ✅ Yes | Attribution | Corpus files have separate terms, but not shipped in the product path. | ✅ (dep) |
| 12 | **mir_eval** | **MIT** | ✅ Yes | Attribution | N/A | eval only |
| 13 | numpy / scipy / pandas / matplotlib / torch / jax / numpyro / soundfile / click / rich / pydantic / tqdm | BSD / Apache-2.0 / MIT family | ✅ Yes | Attribution | N/A | ✅ |

---

## 3. YouTube / audio-sourcing (brief — per your design decision)

You've decided to mirror **Chord AI's operating model**: users supply their own
YouTube links, Harmonia acts as a transformation/analysis tool on user-supplied
URLs, and the platform stores/redistributes no songs itself. That's a
reasonable, industry-common posture and I'm not going to re-litigate it here.

Two honest caveats, then done:

- **This is not a software-license issue at all.** yt-dlp being public-domain
  (row 8) resolves *nothing* about the audio. The exposure is YouTube's **ToS**
  (which prohibits downloading absent an API/permission) and the **underlying
  musical copyright** held by labels/publishers/PROs.
- **User-supplied-URL reduces but does not eliminate exposure.** Pushing the
  "who fetched this" step onto the user is the standard mitigation, but this
  exact business model has, to my knowledge, **not been squarely tested in
  court**, and DMCA/ToS risk is not the same as zero risk. If you later cache,
  store, or re-serve any downloaded audio (vs. transient analysis), the risk
  profile changes materially. Keep the "transient, user-initiated, not stored"
  property as an explicit product invariant, and let counsel bless it.

No further research spent here per your instruction.

---

## 4. Training-data provenance (the often-overlooked category)

Distinct from runtime library licenses: when **model weights that ship in the
product** were trained on a licensed dataset, the dataset's terms can — in
principle — reach the derivative model. In practice most academic MIR datasets
are permissive for *model training* even when they restrict *raw-data
redistribution*, but this varies and is under-litigated. What ships in Harmonia:

- **music-x-lab weights (bundled, row 6):** Humphrey & Bello corpus =
  annotations over McGill Billboard + Isophonics-style sets. The *annotations*
  are academically licensed; the *audio* is commercial and was collected
  privately (not redistributed by the repo). You ship the **weights**, not the
  audio — generally considered acceptable, but untested. **Trust level:
  medium; verify with counsel that you're comfortable shipping weights of
  unknown-per-track audio provenance.**
- **Your in-house weights** (`yt_chord_model*.npz`, `nnls24_heads.npz`,
  `ctx_v3.npz`, `progression_encoder.pt`, etc.): trained on **POP909**
  (research-only license — MIDI/annotations, no commercial redistribution of
  the *dataset*) and **YouTube-sourced audio**. Training a model on
  research-only data and shipping the *weights* is the same gray area as above.
  **Action:** document exactly what each shipped `.npz`/`.pt` was trained on so
  counsel can assess; right now that provenance is not recorded next to the
  files.
- **RWC-Popular / JAAH / GuitarSet / AAM** (named in your task as potential
  training sources): **not found being loaded in the production path.** RWC in
  particular is restrictively licensed (per-institution research agreement, no
  commercial redistribution) — if any shipped weight was trained on RWC audio,
  flag it explicitly to counsel. JAAH/GuitarSet/AAM annotations are
  CC-BY-family and more permissive. **Verify which, if any, fed the shipped
  weights** — I could not confirm from the code that they did.

---

## 5. Recommended actions before launch

1. **madmom (blocker):** either (a) obtain a commercial license from JKU, or
   (b) switch the production default to the librosa beat tracker
   (`--no-madmom` path already exists) and remove madmom from the shipped
   dependency set. Option (b) is the cheap, clean fix — validate the metric
   delta first (CLAUDE.md rule #6: a component swap changes more than the
   target metric — diff beat counts, not just accuracy).
2. **NNLS-Chroma GPL (conditional):** get counsel to confirm your deployment is
   pure server-side SaaS (no plugin binary distributed to users). If you ever
   ship a desktop/offline build, you need a GPL-clean replacement for the NNLS
   front-end (e.g. the Basic-Pitch BP48 feature path that predates it, per
   `chord_pipeline_v1.py` comments).
3. **Demucs/HDemucs:** keep it out of the production build until the MUSDB18
   non-commercial weights question is resolved (retrain on commercially-clear
   data, or license, or don't ship it).
4. **Weights provenance:** write a one-line training-data note next to each
   shipped `.npz`/`.pt` model file so counsel (and future you) can audit.
5. **Attribution/NOTICE:** for the permissive rows (Apache/BSD/MIT/ISC),
   assemble a `THIRD_PARTY_NOTICES` file — cheap, and required by Apache-2.0
   and the MIT/BSD notice clauses.

---

## Sources (verified)

- Basic Pitch — `.venv/.../basic_pitch-0.3.0.dist-info/licenses/LICENSE` (Apache 2.0, "Copyright 2022 Spotify AB") + `NOTICE`.
- librosa — `.venv/.../librosa-0.11.0.dist-info/METADATA` (`License: ISC`).
- madmom — `.venv/.../madmom-0.16.1.dist-info/LICENSE` (full CC BY-NC-SA 4.0 text, incl. Widmer commercial-contact clause) + METADATA (`License: BSD, CC BY-NC-SA`, `Classifier: License :: Free for non-commercial use`).
- NNLS-Chroma / Chordino — GitHub [c4dm/nnls-chroma](https://github.com/c4dm/nnls-chroma), [Isophonics NNLS-Chroma page](https://isophonics.net/nnls-chroma) (GPL v2-or-later, Mauch & Cannam, QMUL C4DM).
- `vamp` wrapper — `.venv/.../vamp-1.1.0.dist-info/METADATA` + `COPYING.rst` (MIT).
- music-x-lab — `harmonia/third_party/ISMIR2019-Large-Vocabulary-Chord-Recognition/LICENSE` (MIT, "Copyright (c) 2023 Music X Lab") + `README.MD`; paper [ISMIR 2019 #78](https://archives.ismir.net/ismir2019/paper/000078.pdf).
- BTC-ISMIR19 — GitHub [jayg996/BTC-ISMIR19](https://github.com/jayg996/BTC-ISMIR19) (MIT; Isophonics/RobbieWilliams/UsPop2002, audio not shipped).
- yt-dlp — `.venv/.../yt_dlp-2026.7.4.dist-info/licenses/LICENSE` (Unlicense / public domain).
- Demucs — `.venv/.../demucs-4.1.0.dist-info/licenses/LICENSE` (MIT, Meta) + METADATA (MUSDB18-HQ training data).
- torchaudio — `.venv/.../torchaudio-2.11.0.dist-info/licenses/LICENSE` (BSD-2-Clause).
- music21 — `.venv/.../music21-10.5.0.dist-info/METADATA` (BSD-3-Clause).

---

*Prepared as engineering due-diligence for the maintainer. Not legal advice.
Two blockers (madmom NC, NNLS-Chroma GPL) and the weights-provenance questions
require review by a qualified IP/entertainment-technology attorney before a
paid launch.*
