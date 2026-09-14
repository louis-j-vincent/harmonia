"""harmonia/server — l'app Flask, en modules par responsabilité (sprints 11-14).

Avant, tout tenait dans `harmonia_min/server.py` (1 774 lignes). Ici : `app.py`
construit l'app et sert les routes statiques, `jobs.py` porte le registre des
tâches d'analyse et `_resolve_audio`, `youtube.py` porte le téléchargement
yt-dlp, et `routes/` porte un fichier par groupe de routes API.

Lancer le serveur : `python -m harmonia.server` (voir `__main__.py`).
"""
