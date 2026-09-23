package fr.harmonia.app

import android.util.Log
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File

/**
 * Le runtime ExecuTorch de Beat This! sur Android (P2d, 2026-09-23).
 *
 * POURQUOI CETTE CLASSE EXISTE. Sur le Mac, `mobile/harmonia_mobile/
 * beats_infer.py` exécute le `.pte` par le runtime PYTHON d'ExecuTorch
 * (`from executorch.runtime import Runtime`). Ce paquet pip n'existe pas sur
 * Android : là-bas le runtime est une bibliothèque NATIVE (`libexecutorch.so`,
 * AAR `org.pytorch:executorch-android`) pilotée par JNI depuis Java/Kotlin.
 * C'est le seul morceau du chemin de calcul qui doit être réécrit hors de
 * Python ; les quatre autres briques (mel, CQT, réseau musx, chroma NNLS)
 * sont du numpy pur et traversent telles quelles.
 *
 * CE QUE LA CLASSE FAIT, ET RIEN DE PLUS : charger le `.pte` une fois, et
 * exécuter `forward` sur un bloc. Elle ne connaît ni le mel-spectrogramme,
 * ni le découpage en blocs de 1500 trames, ni le post-traitement — tout cela
 * reste en numpy dans `beats_infer.py`, identique au Mac, pour qu'un seul
 * code décide des battues sur les deux plateformes.
 *
 * TROIS CHOIX, ET LEUR RAISON :
 *
 * 1. `LOAD_MODE_MMAP`. Le `.pte` de `final0` fait 82 Mo et l'AVD n'a que 2 Go
 *    de RAM. En `LOAD_MODE_FILE` le fichier entier est lu dans le tas du
 *    process ; en mmap, les pages de poids sont mappées depuis le fichier et
 *    le noyau peut les reprendre sous pression mémoire. Les poids sont lus en
 *    boucle par les 7 à 15 blocs d'un morceau, donc ils restent chauds en
 *    pratique.
 * 2. `numThreads = availableProcessors()`. Le troisième paramètre de
 *    `Module.load` règle le pool de threads XNNPACK (vérifié dans la javadoc
 *    de l'AAR 1.4.0, pas supposé). T1c avait noté que ce pool échappe aux
 *    variables `OMP_NUM_THREADS` côté Mac ; ici il est réglé explicitement.
 * 3. Une SEULE sortie aplatie. `forward` rend les deux têtes du réseau
 *    (battues, temps forts) concaténées bout à bout dans un seul `FloatArray`,
 *    parce que c'est ce qui traverse la frontière Chaquopy sans copie
 *    supplémentaire. `mobile_beats_runtime.py` vérifie la longueur totale et
 *    refuse de deviner si elle ne fait pas exactement 2 x le nombre de trames
 *    — un découpage silencieusement faux donnerait des battues plausibles et
 *    fausses, la classe d'erreur n°1 de ce projet.
 *
 * CE QU'ELLE NE RÉSOUT PAS : le modèle n'accepte QUE des blocs de 1500
 * trames (forme figée à l'export, voir `t1c_beat_this_complet.md`). La classe
 * ne le vérifie pas elle-même — c'est le graphe exporté qui refusera une
 * autre forme, avec une erreur du runtime natif, et `beats_infer.split_piece`
 * ne produit jamais autre chose.
 */
object BeatsRuntime {

    private const val TAG = "harmonia.beats"
    private var module: Module? = null
    private var loadedPath: String? = null
    private var threads: Int = 0

    /**
     * Charge le `.pte` (une fois par process). Rappelé avec le MÊME chemin,
     * ne fait rien ; avec un chemin différent, relâche l'ancien module avant
     * de charger le nouveau (sinon deux fois 82 Mo mappés pour rien).
     *
     * @param ptePath chemin FICHIER (pas un asset — les assets Android ne
     *   sont pas des chemins ; `MainActivity.seedModelsOnce` les recopie
     *   d'abord dans `filesDir`).
     * @param numThreads nombre de threads XNNPACK ; <= 0 signifie "autant que
     *   de coeurs".
     */
    @JvmStatic
    @Synchronized
    fun load(ptePath: String, numThreads: Int = 0): Boolean {
        if (module != null && loadedPath == ptePath) return true
        val f = File(ptePath)
        if (!f.exists() || f.length() == 0L) {
            Log.e(TAG, "pte absent ou vide: $ptePath")
            return false
        }
        module?.close()
        module = null
        val n = if (numThreads > 0) numThreads else Runtime.getRuntime().availableProcessors()
        val t0 = System.nanoTime()
        module = Module.load(ptePath, Module.LOAD_MODE_MMAP, n)
        loadedPath = ptePath
        threads = n
        Log.i(TAG, "pte chargé: ${f.length()} octets, $n threads, " +
                "${(System.nanoTime() - t0) / 1_000_000} ms")
        return true
    }

    /** Le nombre de threads réellement demandé au runtime (pour le rapport). */
    @JvmStatic
    fun numThreads(): Int = threads

    /**
     * Un bloc dedans, les deux têtes dehors.
     *
     * @param input le mel-spectrogramme aplati en ordre C (row-major), soit
     *   `shape.product()` valeurs.
     * @param shape la forme, typiquement `[1, 1500, 128]`.
     * @return les sorties du graphe concaténées dans leur ordre de
     *   déclaration : d'abord les logits de battue, puis ceux de temps fort.
     */
    @JvmStatic
    fun forward(input: FloatArray, shape: LongArray): FloatArray {
        val m = module ?: throw IllegalStateException(
            "BeatsRuntime.load() n'a pas été appelé avant forward()")
        var expected = 1L
        for (d in shape) expected *= d
        require(input.size.toLong() == expected) {
            "taille d'entrée ${input.size} != produit de la forme $expected"
        }
        val tensor = Tensor.fromBlob(input, shape)
        val outs: Array<EValue> = m.forward(EValue.from(tensor))
        var total = 0
        val parts = ArrayList<FloatArray>(outs.size)
        for (ev in outs) {
            val a = ev.toTensor().dataAsFloatArray
            parts.add(a)
            total += a.size
        }
        val flat = FloatArray(total)
        var off = 0
        for (a in parts) {
            System.arraycopy(a, 0, flat, off, a.size)
            off += a.size
        }
        return flat
    }

    /** Le nombre de sorties du graphe — sert à la vérification côté Python. */
    @JvmStatic
    fun outputCount(input: FloatArray, shape: LongArray): Int {
        val m = module ?: throw IllegalStateException("load() d'abord")
        val tensor = Tensor.fromBlob(input, shape)
        return m.forward(EValue.from(tensor)).size
    }

    @JvmStatic
    @Synchronized
    fun release() {
        module?.close()
        module = null
        loadedPath = null
    }
}
