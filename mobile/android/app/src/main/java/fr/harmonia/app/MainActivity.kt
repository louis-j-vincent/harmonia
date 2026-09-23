package fr.harmonia.app

import android.annotation.SuppressLint
import android.app.Activity
import android.content.res.AssetManager
import android.os.Bundle
import android.util.Log
import android.webkit.ConsoleMessage
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.FrameLayout
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * Squelette Android P1 (2026-09-23) : une seule Activity, une WebView plein
 * écran, un Flask embarqué (Chaquopy) qui sert `harmonia/static/`. Pas de
 * pipeline ici — voir docs/portage_mobile/p1_squelette_android.md.
 *
 * Ordre au démarrage : 1) copier les 2-3 charts/audio de seed depuis les
 * assets (une fois, au premier lancement — les assets Android ne sont PAS
 * des chemins fichier utilisables par Python) ; 2) démarrer Python +
 * `mobile_entry.start_server` dans un thread ; 3) sonder `/api/library`
 * jusqu'à 200 ; 4) charger la WebView sur `http://127.0.0.1:<port>/`.
 */
class MainActivity : Activity() {

    // 7772 est le port DE PROD SUR LE MAC — interdit là-bas (docs/portage_
    // mobile/PLAN.md). Ici on est DANS l'app Android, sur le téléphone :
    // c'est le même port que le reste du projet utilise pour "l'app",
    // aucune collision possible avec le Mac de Louis.
    private val port = 7772

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val webView = WebView(this)
        setContentView(webView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))

        webView.settings.javaScriptEnabled = true
        webView.settings.domStorageEnabled = true
        webView.settings.mediaPlaybackRequiresUserGesture = false
        webView.settings.allowFileAccess = false
        webView.webViewClient = WebViewClient()
        // Sans ça, une erreur JS (module ES cassé, exception non attrapée…)
        // est invisible depuis `adb logcat` — chromium ne relaie PAS la
        // console par défaut. Demandé après le smoke test P1 (écran blanc,
        // cause trouvée seulement via `logcat -d | grep chromium`).
        webView.webChromeClient = object : WebChromeClient() {
            override fun onConsoleMessage(msg: ConsoleMessage): Boolean {
                Log.i("harmonia.js", "${msg.messageLevel()} ${msg.sourceId()}:${msg.lineNumber()} ${msg.message()}")
                return true
            }
        }

        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        Thread {
            try {
                val (stateDir, audioDir) = seedStateOnce()
                val modelsDir = seedModelsOnce()
                val musxDir = seedMusxOnce()
                val py = Python.getInstance()
                val mobileEntry: PyObject = py.getModule("mobile_entry")
                mobileEntry.callAttr("start_server", stateDir.absolutePath, audioDir.absolutePath,
                    port, modelsDir.absolutePath, musxDir.absolutePath)

                val ready = waitForServer(port, timeoutMs = 20_000)
                Log.i(TAG, "Flask ready=$ready")

                runOnUiThread {
                    webView.loadUrl("http://127.0.0.1:$port/")
                }
            } catch (e: Exception) {
                Log.e(TAG, "startup failed", e)
            }
        }.start()
    }

    /** Copie assets/harmonia_seed (récursif) vers filesDir une seule fois (marqueur
     * `.seeded`). Les assets Android ne sont accessibles qu'via
     * AssetManager — jamais comme chemin fichier direct, d'où la copie. */
    private fun seedStateOnce(): Pair<File, File> {
        val stateDir = File(filesDir, "harmonia_state")
        val audioDir = File(filesDir, "harmonia_audio")
        val marker = File(filesDir, ".seeded")

        if (!marker.exists()) {
            val chartsDir = File(stateDir, "cache/charts").also { it.mkdirs() }
            val humanDir = File(stateDir, "human").also { it.mkdirs() }
            audioDir.mkdirs()

            copyAssetDir("harmonia_seed/charts", chartsDir)
            copyAssetDir("harmonia_seed/audio", audioDir)
            copyAssetFile("harmonia_seed/chart_meta.json", File(humanDir, "chart_meta.json"))
            copyAssetFile("harmonia_seed/folders.json", File(humanDir, "folders.json"))

            marker.writeText("ok")
            Log.i(TAG, "seed copied to $stateDir / $audioDir")
        }
        return Pair(stateDir, audioDir)
    }

    /**
     * Recopie les poids des réseaux (assets/harmonia_models : le `.pte` de
     * Beat This!, 82 Mo, et les 5 `.npz` de musx, 2 Mo chacun) vers
     * `filesDir/harmonia_models`, une fois (P2d, 2026-09-23).
     *
     * POURQUOI COPIER PLUTÔT QUE LIRE L'ASSET. `Module.load` (ExecuTorch) et
     * `np.load` (musx) veulent tous deux un CHEMIN FICHIER ; un asset
     * Android n'en est pas un, c'est une entrée dans le zip de l'APK. Le
     * prix est 92 Mo de stockage privé en plus de l'APK — assumé, et noté
     * dans le rapport.
     *
     * Le marqueur est distinct de `.seeded` (charts/audio) exprès : les
     * poids changent quand on réexporte un modèle, les charts de seed non.
     * Chaque fichier est revérifié par sa TAILLE, pas seulement par sa
     * présence — une copie interrompue laisse un fichier court, et un `.pte`
     * tronqué se manifesterait par une erreur obscure du runtime natif au
     * lieu d'une recopie.
     */
    private fun seedModelsOnce(): File {
        val modelsDir = File(filesDir, "harmonia_models").also { it.mkdirs() }
        val names = assets.list("harmonia_models") ?: emptyArray()
        for (name in names) {
            val dest = File(modelsDir, name)
            // `openFd` ne marche QUE sur un asset non compressé — c'est le
            // cas ici (`noCompress += listOf("pte","npz")` dans
            // app/build.gradle.kts), et ça sert donc aussi de contrôle que ce
            // réglage a bien pris. S'il ne prend pas, on retombe sur le
            // critère faible « le fichier existe et n'est pas vide », en le
            // DISANT dans le log plutôt qu'en le taisant.
            val expected = runCatching {
                assets.openFd("harmonia_models/$name").use { it.length }
            }.getOrElse {
                Log.w(TAG, "asset $name compressé (taille inconnue) — " +
                        "vérification par présence seulement")
                -1L
            }
            val ok = dest.exists() && dest.length() > 0 &&
                    (expected < 0 || dest.length() == expected)
            if (ok) continue
            copyAssetFile("harmonia_models/$name", dest)
            Log.i(TAG, "modèle copié: $name (${dest.length()} octets)")
        }
        return modelsDir
    }

    /**
     * Recopie le clone musx vendu (assets/harmonia_musx) vers
     * `filesDir/harmonia_musx`, une fois (P2d, 2026-09-23).
     *
     * `harmonia/musx.py::redecode` fait `chdir` dans ce dossier et y importe
     * `extractors.xhmm_ismir` : il lui faut de VRAIS fichiers, ce que les
     * assets ne sont pas et ce que l'archive Python de Chaquopy n'est pas non
     * plus. Le chemin obtenu ici est passé à Python, qui le pose dans
     * `HARMONIA_MUSX_DIR`.
     *
     * Le marqueur porte le nombre de fichiers copiés : si le contenu
     * embarqué change un jour (P4, 2026-09-23 : les 27 Mo de `cache_data/
     * *.sdict`, des poids torch que le téléphone n'ouvre jamais, sont déjà
     * exclus — voir `build.gradle.kts::copyMusxClone` et
     * `harmonia/musx.py::_musx_dir`), la recopie se refera d'elle-même au
     * lieu de laisser un dossier à moitié à jour.
     */
    private fun seedMusxOnce(): File {
        val dest = File(filesDir, "harmonia_musx")
        val marker = File(filesDir, ".musx_seeded")
        val n = countAssets("harmonia_musx")
        if (marker.exists() && marker.readText() == n.toString()) return dest
        dest.mkdirs()
        copyAssetTree("harmonia_musx", dest)
        marker.writeText(n.toString())
        Log.i(TAG, "clone musx copié: $dest ($n fichiers)")
        return dest
    }

    private fun countAssets(assetPath: String): Int {
        val names = assets.list(assetPath) ?: return 0
        if (names.isEmpty()) return 1   // c'est un fichier, pas un dossier
        var total = 0
        for (name in names) total += countAssets("$assetPath/$name")
        return total
    }

    /** Copie RÉCURSIVE (contrairement à `copyAssetDir`, qui ne descend pas). */
    private fun copyAssetTree(assetPath: String, dest: File) {
        val names = assets.list(assetPath) ?: return
        if (names.isEmpty()) {
            copyAssetFile(assetPath, dest)
            return
        }
        dest.mkdirs()
        for (name in names) copyAssetTree("$assetPath/$name", File(dest, name))
    }

    private fun copyAssetDir(assetPath: String, dest: File) {
        val am: AssetManager = assets
        val names = am.list(assetPath) ?: return
        for (name in names) {
            copyAssetFile("$assetPath/$name", File(dest, name))
        }
    }

    private fun copyAssetFile(assetPath: String, dest: File) {
        assets.open(assetPath).use { input ->
            FileOutputStream(dest).use { output ->
                input.copyTo(output)
            }
        }
    }

    /** Poll `/api/library` jusqu'à 200 ou timeout. C'est la porte de sortie
     * utilisée dans le rapport : Flask est prêt, la bibliothèque répond. */
    private fun waitForServer(port: Int, timeoutMs: Long): Boolean {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            try {
                val conn = URL("http://127.0.0.1:$port/api/library").openConnection() as HttpURLConnection
                conn.connectTimeout = 500
                conn.readTimeout = 500
                if (conn.responseCode == 200) return true
            } catch (_: Exception) {
                // Flask pas encore levé — on réessaie.
            }
            Thread.sleep(200)
        }
        return false
    }

    companion object {
        private const val TAG = "HarmoniaMain"
    }
}
