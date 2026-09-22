package fr.harmonia.app

import android.annotation.SuppressLint
import android.app.Activity
import android.os.Bundle
import android.util.Log
import android.webkit.CookieManager
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.LinearLayout
import android.widget.Toast
import java.io.File

/**
 * Connexion YouTube pour une session durable sur device (P3c, 2026-09-23).
 * Remplace la procédure manuelle "export cookies sur le Mac + adb push"
 * (P3b), qui rote en <20 min et exige le Mac — voir
 * docs/portage_mobile/p3c_youtube_session.md.
 *
 * Une seule page (accounts.google.com/ServiceLogin), un bouton "Terminé" qui
 * recopie les cookies posés dans CETTE WebView (CookieManager, géré par
 * Android/Google comme une vraie session de navigateur, pas par nous) vers
 * `<filesDir>/youtube_cookies.txt` au format Netscape — le MÊME fichier que
 * `mobile_youtube.py:_cookie_file()` lit déjà, donc rien d'autre à changer
 * côté téléchargement.
 *
 * Lancée par `mobile_youtube.open_login_activity()` (pont Python→Java via
 * Chaquopy `jclass`), appelé par `POST /api/mobile/youtube_login`
 * (mobile_entry.py), lui-même appelé par le bouton "Connect YouTube" de
 * `harmonia/static/screens/prefs.js`.
 *
 * OUVERT (pas vérifié ici, aucun identifiant disponible — voir le rapport) :
 * si les cookies d'AUTHENTIFICATION de Google (SID/HSID/SSID/__Secure-*)
 * sont HttpOnly, `CookieManager.getCookie()` (API native Android, PAS
 * `document.cookie`) est censée quand même les exposer d'après la doc
 * Android — mais ce n'est pas testé de bout en bout ici faute de vrai
 * login. Ce fichier ne marque AUCUN cookie `#HttpOnly_` dans le fichier
 * Netscape (CookieManager ne dit pas lesquels le sont) ; si yt-dlp rejette
 * un cookie essentiel pour cette raison, c'est la prochaine chose à
 * vérifier — avec un vrai compte, PAS ici.
 */
class YouTubeLoginActivity : Activity() {
    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val webView = WebView(this)
        webView.settings.javaScriptEnabled = true
        webView.settings.domStorageEnabled = true
        webView.webViewClient = WebViewClient()

        val doneButton = Button(this).apply {
            text = "Terminé"
            setOnClickListener { saveCookiesAndFinish() }
        }

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            // Android 15 (targetSdk 35) dessine edge-to-edge par défaut : sans
            // ça le bouton "Terminé" (WRAP_CONTENT, ~126px) se retrouve quasi
            // entièrement sous la barre de statut système (128px, mesuré ici
            // via `dumpsys window displays`) — visible, mais les taps dans
            // cette zone sont interceptés par la StatusBar, JAMAIS reçus par
            // le bouton. Trouvé en testant sur l'émulateur (bouton
            // invisible-au-tap malgré des bounds a11y correctes) — voir le
            // rapport, "Ce qui a été vérifié".
            fitsSystemWindows = true
            addView(doneButton, LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
            addView(webView, LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
        }
        setContentView(layout)

        webView.loadUrl("https://accounts.google.com/ServiceLogin?service=youtube")
    }

    /** CookieManager (les deux domaines, fusionnés par nom) -> fichier
     * Netscape (`.youtube.com`, `/`, secure, expiration à 5 ans). Écrit AU
     * MÊME chemin que `mobile_youtube.py` lit déjà — aucun changement côté
     * Python nécessaire pour que le prochain téléchargement l'utilise. */
    private fun saveCookiesAndFinish() {
        val mgr = CookieManager.getInstance()
        val raw = (mgr.getCookie("https://www.youtube.com") ?: "") + "; " +
                  (mgr.getCookie("https://accounts.google.com") ?: "")
        val seen = LinkedHashMap<String, String>()
        raw.split(";").forEach { pair ->
            val idx = pair.indexOf('=')
            if (idx > 0) {
                val name = pair.substring(0, idx).trim()
                val value = pair.substring(idx + 1).trim()
                if (name.isNotEmpty()) seen[name] = value
            }
        }
        val farExpiry = (System.currentTimeMillis() / 1000) + 60L * 60 * 24 * 365 * 5
        val sb = StringBuilder("# Netscape HTTP Cookie File\n")
        for ((name, value) in seen) {
            sb.append(".youtube.com\tTRUE\t/\tTRUE\t$farExpiry\t$name\t$value\n")
        }
        File(filesDir, "youtube_cookies.txt").writeText(sb.toString())
        Toast.makeText(this, "${seen.size} cookies sauvegardés", Toast.LENGTH_SHORT).show()
        Log.i(TAG, "saved ${seen.size} cookie names")
        finish()
    }

    companion object {
        private const val TAG = "harmonia.ytLogin"
    }
}
