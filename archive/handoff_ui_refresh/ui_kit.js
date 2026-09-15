/* ─────────────────────────────────────────────────────────────────────────
   ui_kit.js — the control system for harmonia_min/app_shell.html.

   PASTE THIS BLOCK inside window.APP's IIFE, right after `function haptic(...)`
   (~line 300). It uses el(), T, UI and SERIF, which are already in scope there.
   Do NOT load it as a separate <script>: it depends on the closure.

   Everything the UI refresh asks for comes from these five factories. The rule
   they encode: three sizes (56 primary / 48 tab / 44 everything else), three
   radii (12 / 16 / full), three states (selected / available / off). If a
   control you are writing does not fit one of these, that is a design question
   — ask, don't invent a sixth height.
   ───────────────────────────────────────────────────────────────────────── */

  const SZ = { primary:56, tab:48, control:44, radius:12, radiusLg:16 };

  // A control's three states, as a style fragment. `state`: "on" | "off" | "mute".
  function ctlSkin(state){
    if(state==="on")   return `background:${T.accent};color:#fff;border:1.5px solid ${T.accent};`;
    if(state==="mute") return `background:${T.paper};color:${T.rule};border:1px solid ${T.line};`;
    return `background:${T.card};color:${T.ink};border:1px solid ${T.line};`;
  }

  /* A labelled button. Replaces every ad-hoc `el("button", "...")` in the
     shell — including the ones that carried a symbol prefix ("◎ Set bar 1",
     "⧉ Check merges", "⇧"). Symbols go in `opts.note` as words, not glyphs. */
  function kitButton(label, onClick, opts){
    opts = opts || {};
    const h = opts.height || SZ.control;
    const b = el("button",
      `flex:0 0 auto;height:${h}px;padding:0 ${opts.wide?20:16}px;border-radius:${SZ.radius}px;`+
      `font:600 ${opts.small?13:14}px ${UI};cursor:pointer;display:inline-flex;align-items:center;`+
      `justify-content:center;gap:7px;white-space:nowrap;${ctlSkin(opts.state||"off")}`+
      (opts.grow?"flex:1;":""));
    b.appendChild(el("span","",label));
    if(opts.note) b.appendChild(el("span",
      `font:500 12px ${UI};color:${opts.state==="on"?"rgba(255,255,255,.75)":T.faint};`, opts.note));
    if(onClick) b.onclick=onClick;
    return b;
  }

  /* A round icon button — back, share, prefs. Never smaller than 44. */
  function kitIcon(glyphChar, onClick, aria, opts){
    opts = opts || {};
    const b = el("button",
      `flex:0 0 auto;width:${SZ.control}px;height:${SZ.control}px;border-radius:50%;`+
      `background:${T.card};border:1px solid ${T.line};color:${T.ink};cursor:pointer;`+
      `display:flex;align-items:center;justify-content:center;`+
      `font:${opts.serif?`italic 600 16px ${SERIF}`:`400 22px ${UI}`};line-height:1;`);
    b.textContent = glyphChar;
    if(aria) b.setAttribute("aria-label", aria);
    if(onClick) b.onclick=onClick;
    return b;
  }

  /* The one segmented control. Read/Analyse/Annotate, the Analyse lenses, the
     L1/L2/L3 ladder and the prompter's speed presets are all THIS — no more
     outlined-accent pill rows.
     items: [[value, label], ...]   get(): current value   set(v): commit */
  function kitSegmented(items, get, set, opts){
    opts = opts || {};
    const wrap = el("div",
      `display:flex;gap:6px;background:${T.line};border-radius:${SZ.radiusLg-2}px;padding:4px;`+
      (opts.inline?"":"width:100%;"));
    items.forEach(([v,lb])=>{
      const on = get()===v;
      const b = el("button",
        `flex:${opts.inline?"0 0 auto":"1"};height:${opts.height||SZ.tab}px;`+
        (opts.inline?"padding:0 16px;":"")+
        `border:none;border-radius:${SZ.radius-1}px;cursor:pointer;font:600 15px ${UI};`+
        `background:${on?T.card:"transparent"};color:${on?T.ink:T.faint};`+
        `box-shadow:${on?"0 1px 3px rgba(0,0,0,.13)":"none"};transition:background .15s;`, lb);
      b.onclick=()=>{ haptic(); set(v); };
      wrap.appendChild(b);
    });
    return wrap;
  }

  /* A form/section chip. 44 tall, count as a subordinate span, never a 26px pill. */
  function kitChip(label, count, onClick, active){
    const c = el("button",
      `flex:0 0 auto;height:${SZ.control}px;min-width:62px;padding:0 16px;`+
      `border-radius:${SZ.radius}px;cursor:pointer;display:inline-flex;align-items:center;`+
      `justify-content:center;gap:5px;font:700 16px ${UI};`+
      (active
        ? `border:1.5px solid ${T.accent};background:rgba(138,43,43,.09);color:${T.ink};`
        : `border:1px solid ${T.line};background:${T.card};color:${T.ink};`));
    c.appendChild(el("span","",label));
    if(count>1) c.appendChild(el("span",`font:600 12px ${UI};color:${T.faint};`,"×"+count));
    if(onClick) c.onclick=onClick;
    return c;
  }

  /* One inline legend line — replaces legendFor()'s full sentence row.
     swatch may be null (then only the text shows). */
  function kitLegend(swatch, text){
    const r = el("div",
      `display:flex;align-items:center;gap:8px;margin:9px 4px 0;`+
      `font:italic 13px ${SERIF};color:${T.faint};`);
    if(swatch) r.appendChild(el("div",
      `flex:0 0 auto;width:14px;height:14px;border-radius:4px;background:${swatch};`));
    r.appendChild(el("span","",text));
    return r;
  }
