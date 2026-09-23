# P5 — iOS : prémisses (recherche web, 2026-09-23, avant Xcode)

## Résultat en 3 lignes

**Même architecture, mais avec DEUX ponts neufs sans équivalent Android : verdict provisoire, pas encore un GO.**
Python embarqué + WebView tient (Briefcase/BeeWare, CPython 3.13 officiellement Tier 3 iOS depuis PEP 730 ; `numpy`
n'a pas de roue iOS sur PyPI mais en a une sur le canal `anaconda.org/beeware`, que Briefcase sait consommer).
ExecuTorch a un Swift Package XNNPACK/CoreML qui charge le `.pte` — mais Chaquopy→JNI (Kotlin, déjà écrit) devient
Python→ObjC (`rubicon-objc`, jamais essayé sur `ExecuTorchModule` par personne dont la trace est publique) ou un
shim Swift si `rubicon` ne voit pas l'API. YouTube perd QuickJS-en-sous-processus (impossible, sandbox iOS) et n'a
pas de remplaçant tout tracé : `yt-dlp-ejs` sait parler à Deno/Node/Bun/QuickJS/QuickJS-ng, pas à un runtime fourni
en Python ni à JavaScriptCore — il faudra écrire ce pont. **Estimation, Xcode installé : ≈ 8–12 jours-agent**,
tirée vers le haut par ces deux ponts, pas par le reste (numpy, Flask, yt-dlp purs Python portent tels quels).

## Ce qui a été vérifié (sources, dates)

### Q1 — Python sur iOS (BeeWare/Briefcase)

| Point | Constat | Source |
|---|---|---|
| CPython iOS | Python 3.13 (oct. 2024) = **Tier 3 officiel** iOS (PEP 730) | [peps.python.org/pep-0730](https://peps.python.org/pep-0730/) |
| Roues PyPI iOS | tag accepté depuis pip 24.3 (`cp313-cp313-ios_13_0_arm64_iphoneos` etc.) ; peu de projets en publient encore | [Anaconda blog](https://www.anaconda.com/blog/beeware-mobile-python) |
| **numpy** | **pas de roue iOS sur PyPI** au 2026-09-23 (vérifié : liste des fichiers numpy 2.5.3 sur PyPI, aucun tag `ios`) ; roue existe sur le **canal `anaconda.org/beeware`**, que Briefcase sait ajouter comme index secondaire ; la doc Briefcase cite numpy nommément comme cas à `cleanup_paths` (des `.a` statiques dans la roue à retirer avant soumission App Store) — ce qui implique une roue fonctionnelle, pas juste théorique | [briefcase iOS/xcode docs](https://briefcase.beeware.org/en/stable/reference/platforms/iOS/xcode.html), [PyPI numpy files](https://pypi.org/project/numpy/#files) |
| flask, yt-dlp | purs Python → roue triviale (`py3-none-any`), aucun problème attendu | — |
| Vers 3.15 | `briefcase#3070` (ouverte 2026-09-22) : Python 3.15b1 publie des XCframeworks CPython **officiels** ; Briefcase compte migrer depuis le méta-package `Python-Apple-support` (celui utilisé aujourd'hui) — **pas encore fait** | [briefcase#3070](https://github.com/beeware/briefcase/issues/3070) |
| Alternatives | `kivy-ios` : toolchain mature, compile numpy lui-même (recette communautaire ancienne, fonctionne) mais réimplique tout le pont UI en Kivy, pas WebView — hors cible. `Pyto` : app App-Store, IDE Python avec numpy iOS embarqué, mais **pas embarquable** dans une app tierce — inutilisable ici. | [kivy-ios](https://github.com/kivy/kivy-ios) |

### Q2 — ExecuTorch sur iOS

- Distribué en **Swift Package** (pas d'AAR) : cibles `executorch`, `backend_xnnpack`, `backend_coreml`, `backend_mps`, `kernels_*`. [docs iOS](https://docs.pytorch.org/executorch/stable/using-executorch-ios.html)
- Compatibilité `.pte` : même logique qu'Android (schéma `ET12`, compat descendante) — **non revérifié pour la version Swift PM actuelle** (pas trouvé son numéro exact ; à traiter comme Android : peut être en retard sur le pip `1.5.1` comme l'AAR l'était).
- **Simulateur ≠ émulateur Android.** Le simulateur iOS sur Apple Silicon tourne en arm64 natif (pas d'hyperviseur qui laisse fuiter des bits SME sans activer l'état, comme sur l'émulateur Android) — devrait donc exécuter XNNPACK réellement, contrairement au blocage `rdsvl`/SIGILL trouvé en P2d. Un bug SME2 **différent** existe bien sur Apple Silicon réel (`XNNPACK#9898`, 2026-04-07, segfault sur convs int8 quantifiées via `kai_run_lhs_imatmul_pack…sme`, réglable par `XNNPACK_ENABLE_ARM_SME2=OFF`) — mais c'est **fp32 qui marche**, et nos réseaux (`beat_this`, `musx`) sont fp32.

### Q3 — Pont Python ↔ iOS

- `rubicon-objc` (projet BeeWare officiel, déjà dans le template Briefcase iOS) donne un accès générique à **n'importe quelle classe ObjC** — en principe `AVAudioFile`/`AVAssetReader` (décodage PCM) ET `ExecuTorchModule` (si exposée `@objc`) sont atteignables sans Swift. Aucune recette publiée trouvée pour ni l'un ni l'autre cas précis (la doc rubicon ne montre qu'`AVAudioPlayer`, pas d'extraction PCM) — **à vérifier**, pas confirmé.
- Risque identifié : si `ExecuTorchModule` (Swift) n'expose pas ses symboles à l'exécution ObjC (`@objc` manquant), `rubicon` ne le verra pas → repli sur un shim Swift minimal (quelques lignes) qui, lui, appelle l'API documentée (`ExecuTorchModule.initWithFilePath:`, `loadMethod:error:`, `forwardWithInputs:error:`). [ExecuTorch iOS API](https://docs.pytorch.org/executorch/stable/using-executorch-ios.html)
- WKWebView + Flask localhost : **pas de blocage ATS** si on cible l'IP littérale `127.0.0.1` (comme sur Android) — seul le nom `localhost` demanderait une exception `NSExceptionDomains` dans `Info.plist`. Confirmé par plusieurs sources indépendantes sur le comportement ATS.

### Q4 — YouTube sur iOS

Classé par faisabilité (aucune n'est un simple réglage) :

1. **WKWebView `evaluateJavaScript:` comme runtime JS** — faisable techniquement (l'API existe, retourne des valeurs), mais `yt-dlp-ejs` n'a pas de point d'extension documenté pour un runtime "callback" — il faudrait écrire soi-même le pont qui nourrit le challenge `n` au JS de la page et relit le résultat. Le plus réaliste, mais **à écrire en entier**.
2. **QuickJS lié en bibliothèque C** — pas de roue Python iOS pour `quickjs`/`quickjs-ng`/`quickjs-rs` (vérifié sur PyPI). QuickJS est un petit C mono-fichier, compilable en framework iOS et appelable via `rubicon`/`ctypes` — faisable mais tout le liage est à faire, rien à réutiliser.
3. **`yt-dlp-ejs` avec un runtime "fourni en Python"** — non confirmé : les runtimes documentés (Deno, Node, Bun, QuickJS, QuickJS-ng) sont tous des **processus externes** invoqués par sous-processus ; rien ne dit qu'un objet Python puisse s'y substituer in-process. À vérifier sur le dépôt `yt-dlp/ejs`, pas fait ici (hors budget web).
4. **QuickJS en sous-processus séparé (la solution Android)** — **impossible** : l'exécution d'un binaire arbitraire est bloquée par le bac à sable iOS, sideload compris, pas seulement une règle App Store.

JavaScriptCore (moteur JS natif iOS, `JSContext`) n'est mentionné nulle part dans la doc `yt-dlp-ejs` — il rejoindrait l'option 1 (pont à écrire), pas une option prête à l'emploi.

### Q5 — Distribution perso

- Apple ID gratuit : **3 apps actives, re-signature tous les 7 jours** ; AltStore/Sideloadly automatisent la re-signature via un Mac/PC qui tourne (AltServer en fond, en Wi-Fi). Compte payant (99 $/an, fait connu, non re-sourcé ici) → TestFlight, builds valables 90 jours.
- Xcode : Command Line Tools seules ≈ **1,2–1,8 Go** ; Xcode complet (2026, ~16.x) **≈ 13 Go** en tarball App Store, mentions jusqu'à **40 Go+** une fois simulateurs/plateformes additionnels installés. Avec 5 Go libres actuels sur la machine, **installation bloquée en l'état** — il faudra libérer de l'espace avant même de commencer Q1-Q4 en pratique.
- Briefcase iOS a besoin d'`xcodebuild` (génération + build du projet Xcode) — les Command Line Tools seules ne suffisent pas ; ce point est la pratique standard Apple, pas re-sourcé précisément cette session, à confirmer avec `briefcase create iOS` une fois Xcode posé.

### Q6 — Ce qui ne se réutilise pas

| Brique Android (Kotlin) | Équivalent iOS | Langage |
|---|---|---|
| `AudioDecoder.kt` (MediaCodec → PCM) | `AVAudioFile`/`AVAssetReader` | Swift, ou Python via `rubicon` (non vérifié) |
| `BeatsRuntime.kt` (JNI `Module.load`) | `ExecuTorchModule` | Swift, ou Python via `rubicon` (non vérifié) |
| `YouTubeLoginActivity.kt` (WebView login) | `WKWebView` dans un écran natif | Swift/ObjC, ou Python via `rubicon` |
| `MainActivity.kt` (seed assets, entrée app) | `main.py` de Briefcase (scaffold généré) | **reste Python** — Chaquopy disparaît, remplacé par le runtime Briefcase |
| Chaquopy (le pont lui-même) | `Python-Apple-support` (framework CPython embarqué) | n/a — mécanisme différent, pas un port |
| QuickJS sous-processus | JS in-process (WKWebView ou QuickJS lié) | à écrire, cf. Q4 |
| Gradle | Projet Xcode (généré/piloté par Briefcase) | n/a |
| `nnls_chroma.py`, `musx_net.py`, mel `beat_this`, post-traitement (numpy pur) | **identiques** | Python — **0 portage** |

## Ce qui reste ouvert — 3 prémisses à trancher en <1h chacune, une fois Xcode installé

1. **`rubicon-objc` voit-il `ExecuTorchModule` ?** Dans une app Briefcase iOS hello-world liant le Swift Package
   ExecuTorch, `ObjCClass('ExecuTorchModule')` résout-il, et `loadMethod:error:`/`forwardWithInputs:error:`
   s'appellent-ils depuis Python sans Swift ? Si non → shim Swift minimal, budget révisé à la hausse.
2. **Le `.pte` XNNPACK (`ET12`, exporté par `executorch` 1.5.1) tourne-t-il sur le Simulateur iOS (Apple Silicon,
   arm64 natif) sans le SIGILL SME vu sur l'émulateur Android ?** Un seul `forward()` sur `beat_this` suffit à
   trancher si le simulateur se comporte comme un vrai téléphone ou comme l'émulateur Android cassé.
3. **`numpy` s'importe-t-il et calcule-t-il** (`import numpy; np.array([1,2,3]).sum()`) dans une app Briefcase iOS
   qui déclare la dépendance via le canal `anaconda.org/beeware`, sur simulateur ET sur device de test — avant de
   se soucier du nettoyage `cleanup_paths` (qui ne compte que pour la soumission App Store, pas le sideload).

Non tranché non plus, hors budget de cette recherche : le point d'extension in-process de `yt-dlp-ejs` (Q4.3), et
la version exacte du Swift Package ExecuTorch disponible aujourd'hui vs le schéma `.pte` `ET12`.

## Fichiers produits

* `docs/portage_mobile/p5_ios_premisses.md` — ce rapport. Recherche web + lecture locale uniquement, aucun code
  touché, aucune installation.
