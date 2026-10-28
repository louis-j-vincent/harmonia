// Écran Réglages (2026-10-28, demande de Louis après le switch de modèle de
// battues de la veille : « une interface réglages générale sans aller sur
// une musique en particulier »). Pas de sous-écran par réglage pour
// l'instant — un seul existe (modèle de battues par défaut,
// `beat_editor.js::mountBeatModelPicker`) ; d'autres s'ajouteront comme des
// sections de plus dans la même liste, pas comme des écrans séparés, tant
// qu'il n'y en a qu'une poignée.

import { appBar, screenWrap } from "../router.js";
import { mountBeatModelPicker } from "./beat_editor.js";
import { SERIF, T, UI, el } from "../ui/kit.js";

export function renderSettings(){
  const w = screenWrap();
  w.appendChild(appBar("Réglages", null, "library"));

  const body = el("div",
    `flex:1;min-height:0;overflow-y:auto;padding:18px 20px calc(18px + env(safe-area-inset-bottom));`+
    `display:flex;flex-direction:column;gap:6px;`);
  body.className = "ap-scroll";

  body.appendChild(el("div",
    `font:700 11px ${UI};letter-spacing:.08em;text-transform:uppercase;color:${T.faint};`,
    "Modèle de battues par défaut"));
  body.appendChild(el("div",
    `font:italic 13px ${SERIF};color:${T.faint};line-height:1.5;margin-bottom:10px;`,
    "s'applique à toute nouvelle analyse sur ce téléphone, y compris après un redémarrage de l'appli"));

  const status = el("div", `font:600 12px ${UI};color:${T.faint};margin-bottom:8px;`, "chargement…");
  body.appendChild(status);
  const list = el("div", "display:flex;flex-direction:column;gap:8px;");
  body.appendChild(list);
  mountBeatModelPicker(list, status);

  w.appendChild(body);
  return w;
}
