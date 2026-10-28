// Filet de sécurité mobile (2026-10-27, demande de Louis après le premier
// test de vitesse Android) : corriger la grille de battues à la main quand
// un modèle rapide (small0/int8, voir openBeatModelDebug ci-dessous) se
// trompe, et comparer les modèles de battues À L'OREILLE sans rebuild d'APK.
//
// Reste MINIMAL par construction (demande explicite) : pas de rendu de forme
// d'onde, pas de transport synchronisé comme openBar1Sheet
// (sections_editor.js) — une frise de marqueurs, taper pour choisir/ajouter,
// un bouton pour supprimer, un second tap pour déplacer. Suffisant pour un
// filet de sécurité, pas un éditeur de battues complet.

import { api } from "../api.js";
import { go } from "../router.js";
import { paintLoading, pollJob } from "../screens/analyse.js";
import { S } from "../state.js";
import { SERIF, SZ, T, UI, clear, closeOverlay, el, handle, haptic, overlay, sheetCol, toast } from "../ui/kit.js";

const root = document.getElementById("app");

async function _runJobFor(m, postPath, body){
  S.job = {title: m.title, stage: 0, phase: "listening", error: null, file: null};
  go("analysing");
  try {
    const r = await api.post(postPath, body || {});
    if (r.error) { S.job.error = r.error; paintLoading(); return; }
    pollJob(r.job_id);
  } catch (e) { S.job.error = String(e); paintLoading(); }
}

// ── Éditeur de battues ──────────────────────────────────────────────────

export function openBeatEditor(){
  const m = S.model; if (!m || !m.file) return;
  let beats = (m.beatTimes || []).slice().sort((a, b) => a - b);
  let downbeats = (m.barGrid || []).slice().sort((a, b) => a - b);
  if (beats.length < 4) { toast("pas assez de battues à corriger"); return; }
  const dur = beats[beats.length - 1] + 1.5;
  const PXS = 70; // pixels par seconde — assez large pour taper une battue précise

  let mode = "add";     // "add" | "move"
  let selected = -1;    // index dans `beats`, ou -1

  const back = overlay(); const sheet = sheetCol(); sheet.appendChild(handle());
  const close = () => closeOverlay(back);
  sheet.appendChild(el("div", `text-align:center;font:700 15px ${UI};color:${T.ink};margin-bottom:2px;`,
    "Corriger les battues"));
  sheet.appendChild(el("div", `text-align:center;font:italic 12.5px ${SERIF};color:${T.faint};line-height:1.5;margin-bottom:10px;`,
    "touche une battue pour la choisir · touche la frise pour en ajouter une"));

  const modeRow = el("div", "display:flex;gap:8px;flex:0 0 auto;margin-bottom:10px;");
  sheet.appendChild(modeRow);

  const strip = el("div",
    `position:relative;height:84px;background:${T.paper};border:1px solid ${T.line};`+
    `border-radius:14px;overflow-x:auto;overflow-y:hidden;flex:0 0 auto;touch-action:pan-x;`);
  const track = el("div", "position:relative;height:100%;");
  strip.appendChild(track);
  sheet.appendChild(strip);

  const info = el("div", `text-align:center;font:600 12px ${UI};color:${T.faint};margin-top:9px;min-height:30px;flex:0 0 auto;`, "");
  sheet.appendChild(info);

  function modeBtn(label, active, onClick){
    const b = el("button",
      `flex:1;height:${SZ.control}px;border-radius:${SZ.radius}px;border:1px solid ${T.line};`+
      `font:600 13px ${UI};cursor:pointer;background:${active ? T.accent : T.card};`+
      `color:${active ? "#fff" : T.ink};`, label);
    b.onclick = onClick;
    return b;
  }

  function deleteSelected(){
    if (selected < 0) return;
    const t = beats[selected];
    beats.splice(selected, 1);
    downbeats = downbeats.filter(d => Math.abs(d - t) > 1e-6);
    selected = -1;
    haptic();
    refresh();
  }

  function refresh(){
    clear(modeRow);
    modeRow.appendChild(modeBtn("Ajouter", mode === "add", () => { mode = "add"; refresh(); }));
    modeRow.appendChild(modeBtn("Déplacer", mode === "move", () => { mode = "move"; refresh(); }));
    const del = modeBtn("Supprimer", false, deleteSelected);
    if (selected < 0) { del.style.opacity = ".4"; del.disabled = true; }
    else { del.style.background = T.card; del.style.color = T.ink; }
    modeRow.appendChild(del);

    track.style.width = Math.round(dur * PXS + 40) + "px";
    clear(track);
    downbeats.forEach(t => track.appendChild(el("div",
      `position:absolute;top:6px;bottom:6px;left:${(t * PXS - 1).toFixed(1)}px;width:2px;background:${T.accent};border-radius:1px;`)));
    beats.forEach((t, i) => {
      const hit = el("div",
        `position:absolute;top:0;bottom:0;left:${(t * PXS - 11).toFixed(1)}px;width:22px;cursor:pointer;`);
      const sel = i === selected;
      hit.appendChild(el("div",
        `position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);`+
        `width:${sel ? 15 : 9}px;height:${sel ? 15 : 9}px;border-radius:50%;`+
        `background:${sel ? T.accent : T.ink};opacity:${sel ? 1 : .55};`));
      hit.onclick = (e) => { e.stopPropagation(); haptic(); selected = (selected === i) ? -1 : i; refresh(); };
      track.appendChild(hit);
    });
    info.textContent = `${beats.length} battues · ${downbeats.length} temps forts`+
      (selected >= 0 ? ` · sélectionnée à ${beats[selected].toFixed(2)} s — tape la frise pour la déplacer` : "");
  }

  track.onclick = (e) => {
    const r = track.getBoundingClientRect();
    const t = Math.round(Math.max(0, (e.clientX - r.left) / PXS) * 10000) / 10000;
    if (mode === "add") {
      beats.push(t); beats.sort((a, b) => a - b);
      haptic(); refresh();
    } else if (mode === "move" && selected >= 0) {
      beats[selected] = t; beats.sort((a, b) => a - b);
      selected = beats.indexOf(t);
      haptic(); refresh();
    }
  };

  refresh();

  const apply = el("button",
    `flex:0 0 auto;width:100%;margin-top:14px;height:${SZ.primary}px;border:none;border-radius:${SZ.radius}px;`+
    `background:${T.accent};color:#fff;font:700 14.5px ${UI};cursor:pointer;`, "Valider la correction");
  apply.onclick = () => {
    if (beats.length < 4) { toast("il faut au moins 4 battues"); return; }
    close();
    _runJobFor(m, "/api/beats/" + m.file, { beats, downbeats });
  };
  sheet.appendChild(apply);

  const resetBtn = el("button",
    `flex:0 0 auto;width:100%;margin-top:6px;background:transparent;border:none;color:${T.faint};`+
    `font:600 12.5px ${UI};padding:8px;min-height:${SZ.control}px;cursor:pointer;`,
    "Oublier la correction (revenir au traceur)");
  resetBtn.onclick = () => { close(); _runJobFor(m, "/api/beats/" + m.file + "/reset"); };
  sheet.appendChild(resetBtn);

  const cancel = el("button",
    `flex:0 0 auto;width:100%;margin-top:6px;background:transparent;border:none;color:${T.faint};`+
    `font:600 13px ${UI};padding:8px;min-height:${SZ.control}px;cursor:pointer;`, "Fermer");
  cancel.onclick = close;
  sheet.appendChild(cancel);

  back.appendChild(sheet); root.appendChild(back);
}

// ── Switch de modèle de battues (debug, T5 2026-10-27) ──────────────────
//
// Visible seulement derrière `?debug=1` (S.debugMobile, posé par router.js
// au démarrage) — jamais dans le chemin normal de l'app, exactement la
// même discrétion que `&edit=sections`. Recharge `BeatsRuntime` côté
// Android (mobile_beatmodel.py) puis laisse Louis relancer l'analyse
// lui-même pour écouter le résultat.

const MODEL_NAMES = {
  final0: "final0 — production, 82 Mo",
  small0: "small0 — 10x plus petit, 10 Mo",
  int8: "int8 — quantifié, 21 Mo",
};

export function openBeatModelDebug(){
  const back = overlay(); const sheet = sheetCol(); sheet.appendChild(handle());
  const close = () => closeOverlay(back);
  sheet.appendChild(el("div", `text-align:center;font:700 15px ${UI};color:${T.ink};margin-bottom:2px;`,
    "Modèle de battues (debug)"));
  sheet.appendChild(el("div", `text-align:center;font:italic 12.5px ${SERIF};color:${T.faint};line-height:1.5;margin-bottom:12px;`,
    "bascule à chaud, puis relance l'analyse pour comparer à l'oreille"));

  const status = el("div", `text-align:center;font:600 12px ${UI};color:${T.faint};margin-bottom:10px;`, "chargement…");
  sheet.appendChild(status);
  const list = el("div", "display:flex;flex-direction:column;gap:8px;flex:0 0 auto;");
  sheet.appendChild(list);

  async function refresh(){
    let r;
    try { r = await api.get("/api/debug/beat_model"); }
    catch (e) { status.textContent = "serveur injoignable (" + e + ")"; return; }
    clear(list);
    Object.keys(MODEL_NAMES).forEach(name => {
      const present = !!(r.present_on_device && r.present_on_device[name]);
      const active = r.current === name;
      const b = el("button",
        `display:block;width:100%;text-align:left;padding:12px 14px;border-radius:${SZ.radius}px;`+
        `border:1px solid ${T.line};font:600 13.5px ${UI};cursor:pointer;`+
        `background:${active ? T.accent : T.card};color:${active ? "#fff" : (present ? T.ink : T.faint)};`,
        MODEL_NAMES[name] + (active ? " — actif" : "") + (present ? "" : " — absent sur l'appareil"));
      if (present && !active) {
        b.onclick = async () => {
          status.textContent = "chargement du modèle…";
          try {
            const rr = await api.post("/api/debug/beat_model", { name });
            if (rr.error) { toast(rr.error); status.textContent = ""; refresh(); return; }
            toast(name + " chargé · cache vidé (" + rr.cleared_cache_files + " fichier(s)) — relance l'analyse");
            refresh();
          } catch (e) { toast("échec: " + e); refresh(); }
        };
      } else { b.disabled = true; b.style.cursor = "default"; }
      list.appendChild(b);
    });
    status.textContent = "actuellement chargé : " + r.current;
  }
  refresh();

  const cancel = el("button",
    `flex:0 0 auto;width:100%;margin-top:14px;background:transparent;border:none;color:${T.faint};`+
    `font:600 13px ${UI};padding:8px;min-height:${SZ.control}px;cursor:pointer;`, "Fermer");
  cancel.onclick = close;
  sheet.appendChild(cancel);

  back.appendChild(sheet); root.appendChild(back);
}
