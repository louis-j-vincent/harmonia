// Les dossiers de la bibliothèque, dans leur propre module.
//
// Ils vivaient dans `screens/library.js`. L'écran d'ANALYSE en a besoin depuis
// le 2026-09-18 — Louis : « dès qu'on crée un nouveau chart, pendant que
// l'outil cherche les accords, tu peux déjà me proposer l'option : dans quel
// folder le ranger ? » — et `library.js` importe déjà `analyse.js`, donc les
// y laisser aurait fait un cycle d'imports entre deux écrans. Une notion
// partagée par deux écrans n'appartient à aucun des deux.
//
// Un chart appartient à AU PLUS un dossier (c'est un classeur, pas des tags) :
// « ranger » vide « Hors dossier ». Le localStorage est la source de vérité,
// POST /api/folders en write-through (le serveur en garde une copie).

import { api } from "./api.js";
import { S } from "./state.js";

export const FOLDERS_KEY="harmFolders";

export function loadFolders(){
    try{ return JSON.parse(localStorage.getItem(FOLDERS_KEY))||{order:[],of:{}}; }
    catch(e){ return {order:[],of:{}}; }
  }

export function saveFolders(f){
    try{ localStorage.setItem(FOLDERS_KEY, JSON.stringify(f)); }catch(e){}
    api.post("/api/folders", f).catch(()=>{});
  }

export function folderCount(name){
    const f=S.folders||loadFolders();
    return (S.library||[]).filter(c=>f.of[c.file]===name).length;
  }

// Ranger un chart, sans passer par l'écran de la bibliothèque.
export function ranger(chartFile, dossier){
    if(!chartFile) return;
    const f=loadFolders();
    if(dossier){
      if(!(f.order||[]).includes(dossier)) (f.order=f.order||[]).push(dossier);
      f.of[chartFile]=dossier;
    } else { delete f.of[chartFile]; }
    S.folders=f; saveFolders(f);
  }

/* La copie serveur, FUSIONNÉE et jamais imposée (2026-09-18). Le localStorage
   reste la source de vérité, mais il est PAR APPAREIL : un classement fait sur
   le Mac n'existait pas sur l'iPhone, et un navigateur vidé le perdait. On ne
   reprend de la copie que les charts que cet appareil ne classe PAS déjà — un
   classement local gagne toujours. C'est aussi ce qui fait apparaître
   « Old charts » sur son téléphone sans qu'il range cinquante morceaux. */
export async function fusionnerDossiersServeur(){
    let d; try{ d=await api.get("/api/folders"); }catch(e){ return false; }
    const f=loadFolders(); let bouge=false;
    for(const nom of (d.order||[]))
      if(!(f.order||[]).includes(nom)){ (f.order=f.order||[]).push(nom); bouge=true; }
    for(const [chart,nom] of Object.entries(d.of||{}))
      if(!f.of[chart]){ f.of[chart]=nom; bouge=true; }
    if(bouge){ S.folders=f; saveFolders(f); }
    return bouge;
  }
