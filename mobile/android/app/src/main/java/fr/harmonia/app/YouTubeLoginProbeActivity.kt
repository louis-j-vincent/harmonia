package fr.harmonia.app

import android.annotation.SuppressLint
import android.app.Activity
import android.os.Bundle
import android.util.Log
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.FrameLayout

/**
 * Sonde P3c (2026-09-23), PAS l'écran de connexion final : teste juste,
 * sans identifiants, si une WebView app chargeant
 * accounts.google.com/ServiceLogin?service=youtube affiche le formulaire
 * de connexion ou le mur "ce navigateur n'est peut-être pas sécurisé".
 * Lancée via `adb shell am start -n fr.harmonia.app/.YouTubeLoginProbeActivity
 * --es ua chrome` (extra "ua" = "default" | "chrome"). Voir
 * docs/portage_mobile/p3c_youtube_session.md.
 */
class YouTubeLoginProbeActivity : Activity() {
    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val webView = WebView(this)
        setContentView(webView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))

        webView.settings.javaScriptEnabled = true
        webView.settings.domStorageEnabled = true

        val ua = intent.getStringExtra("ua") ?: "default"
        if (ua == "chrome") {
            // UA Chrome mobile Android courant (2026), pour voir si Google
            // se contente de sniffer l'UA plutôt que de vraies capacités JS.
            webView.settings.userAgentString =
                "Mozilla/5.0 (Linux; Android 15; Pixel 8) AppleWebKit/537.36 " +
                "(KHTML, like Gecko) Chrome/128.0.0.0 Mobile Safari/537.36"
        }
        Log.i(TAG, "probe ua=$ua effectiveUA=${webView.settings.userAgentString}")

        webView.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView?, url: String?) {
                Log.i(TAG, "onPageFinished url=$url")
            }
        }
        webView.loadUrl("https://accounts.google.com/ServiceLogin?service=youtube")
    }

    companion object {
        private const val TAG = "harmonia.ytLoginProbe"
    }
}
