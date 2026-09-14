// Harmonia app shell — module entry point (sprint 16-19 frontend split).
// Was the tail of harmonia_min/app_shell.html's single inline <script>:
//   window.APP = (function(){ ... return API; })();
//   window.APP.build(document.getElementById("app"));
// `API` (with `.build` attached) now lives in router.js, alongside `go()` and
// the deep-link parsing it does at startup — see that file's own comments.
// Nothing outside this app reads anything off `window.APP` except `.build`
// (checked: `grep -n "APP\." harmonia_min/app_shell.html` and docs/plots/*.html
// both come up empty beyond this same call), so this is the whole surface.
import { API } from "./router.js";

window.APP = API;
window.APP.build(document.getElementById("app"));
