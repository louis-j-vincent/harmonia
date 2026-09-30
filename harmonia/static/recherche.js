// La recherche GLOBALE de « Mes charts » (2026-10-01, Louis : « quand je
// cherche un chart, il faut déjà que je sois dans le bon dossier, c'est
// relou »). Module pur — aucune dépendance au DOM ni à l'état — pour être
// testé sous node (tests/test_recherche_globale.py). Le look et la règle
// sont écrits dans docs/STATE.md, « La bibliothèque ».
//
// Ce que ça NE fait PAS : pas de tolérance aux fautes de frappe (« autum »
// trouve « Autumn », « autunm » ne trouve rien), et pas de recherche dans les
// accords ni dans les noms de dossiers.

// minuscules, sans accents, ponctuation → espace (« Beyoncé » → « beyonce »)
export function plier(s){
  return String(s||"").normalize("NFD").replace(/[̀-ͯ]/g,"")
    .toLowerCase().replace(/[^a-z0-9]+/g," ").trim();
}

// Qualité d'une correspondance (plus petit = meilleur), ou null si un des
// mots tapés n'apparaît nulle part dans artiste + titre.
//   0 titre qui commence par la requête · 1 un mot du titre commence par elle
//   2 l'artiste commence par elle / un de ses mots · 3 n'importe où
export function rang(name, q){
  const Q=plier(q); if(!Q) return 0;
  const T=plier(name.title), A=plier(name.artist), tout=A+" "+T;
  if(!Q.split(" ").every(m=>tout.indexOf(m)>=0)) return null;
  if(T.startsWith(Q)) return 0;
  if((" "+T).indexOf(" "+Q)>=0) return 1;
  if((" "+A).indexOf(" "+Q)>=0) return 2;
  return 3;
}

// « Jazz/Standards » → « Jazz › Standards » ; pas de dossier → « Hors dossier »
export function cheminDossier(nom){
  if(!nom) return "Hors dossier";
  return String(nom).split("/").map(s=>s.trim()).filter(Boolean).join(" › ");
}

// Tous les charts de la bibliothèque, quel que soit leur dossier, qui
// correspondent à `q` — liste plate classée par rang puis par titre.
// `nomDe(c)` rend {artist,title} (chartName de library.js), `of` est la table
// fichier → dossier. Chaque résultat : {chart, dossier, rang}.
export function chercherPartout(library, of, q, nomDe){
  const out=[];
  (library||[]).forEach(c=>{
    const n=nomDe(c), r=rang(n,q);
    if(r===null) return;
    out.push({chart:c, dossier:cheminDossier((of||{})[c.file]), rang:r,
              _cle:plier(n.title)+" "+plier(n.artist)});
  });
  out.sort((a,b)=>a.rang-b.rang || (a._cle<b._cle?-1:a._cle>b._cle?1:0));
  return out.map(({_cle,...x})=>x);
}
