package fr.harmonia.app

import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Décodeur audio Android — remplace `ffmpeg -ac 1 -ar <sr>` / `librosa.load`
 * / `soundfile` du pipeline Mac (P2a, 2026-09-23). Contrat exact que ce
 * fichier DOIT respecter : docs/portage_mobile/p0_t3_deps_natives.md,
 * section "Le contrat de décodage audio" ; les chiffres mesurés contre le
 * Mac sont dans docs/portage_mobile/p2a_decodeur_audio.md.
 *
 * `decode(path, targetSr)` : `MediaExtractor` démuxe, `MediaCodec` décode
 * (PCM float si le codec le rend, sinon PCM 16 bits converti), downmix mono
 * par MOYENNE SIMPLE des canaux (= `librosa.core.to_mono`, PAS un mixage
 * pondéré — vérifié dans `librosa/core/audio.py`), retire le délai
 * d'amorçage AAC (« encoder delay/padding », lus depuis les clés de format
 * `MediaFormat` que le `MPEG4Extractor` d'Android remplit à partir de la
 * boîte MP4 `elst` — measured needed, voir p2a), puis rééchantillonne au SR
 * demandé par un FIR polyphase à fenêtre de Kaiser (implémentation directe,
 * pas de FFT). PAS `android.media.audiofx`/`Resampler` — API privée,
 * inaccessible à une app.
 *
 * Un `object` Kotlin plutôt qu'une classe : pas d'état entre deux décodages,
 * et `@JvmStatic` pour que `mobile_audio.py` (Chaquopy) puisse appeler
 * `AudioDecoder.decode(...)` directement, sans passer par `.INSTANCE`.
 */
object AudioDecoder {

    private const val TAG = "AudioDecoder"

    // Clés MediaFormat non exposées comme constantes publiques nommées dans
    // le SDK mais documentées et lues par le framework (MPEG4Extractor,
    // ExoPlayer) depuis la boîte MP4 `elst` — voir p2a pour la mesure qui a
    // confirmé (ou infirmé) qu'elles arrivent bien remplies sur ces fichiers.
    private const val KEY_ENCODER_DELAY = "encoder-delay"
    private const val KEY_ENCODER_PADDING = "encoder-padding"

    @JvmStatic
    fun durationSeconds(path: String): Double {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(path)
            val format = findAudioTrack(extractor) ?: return 0.0
            return if (format.containsKey(MediaFormat.KEY_DURATION))
                format.getLong(MediaFormat.KEY_DURATION) / 1_000_000.0
            else 0.0
        } finally {
            extractor.release()
        }
    }

    /** PCM float32 mono au SR demandé — le seul point d'entrée depuis Python. */
    @JvmStatic
    fun decode(path: String, targetSr: Int): FloatArray {
        val (mono, nativeSr) = decodeNativeMono(path)
        return if (nativeSr == targetSr) mono else resample(mono, nativeSr, targetSr)
    }

    // ───────────────────────── démuxage + décodage ─────────────────────────

    private fun findAudioTrack(extractor: MediaExtractor): MediaFormat? {
        for (i in 0 until extractor.trackCount) {
            val format = extractor.getTrackFormat(i)
            val mime = format.getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("audio/")) {
                extractor.selectTrack(i)
                return format
            }
        }
        return null
    }

    private fun decodeNativeMono(path: String): Pair<FloatArray, Int> {
        val extractor = MediaExtractor()
        extractor.setDataSource(path)
        val format = findAudioTrack(extractor)
            ?: throw IllegalArgumentException("AudioDecoder: no audio track in $path")

        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val channelCount = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        // MESURÉ (docs/portage_mobile/p2a_decodeur_audio.md) : sur ce
        // décodeur logiciel AAC (émulateur API 35), la sortie PCM brute de
        // MediaCodec a DÉJÀ retiré le délai d'amorçage — appliquer
        // KEY_ENCODER_DELAY par-dessus décale tout le signal de 1024
        // échantillons EN RETARD (vérifié par corrélation croisée contre le
        // Mac : décalage exact de -1024 avec le trim naïf, 0 sans). Ces deux
        // clés sont donc lues pour le journal/diagnostic mais PAS utilisées
        // pour couper le PCM — voir `expectedFrames` ci-dessous, qui calcule
        // la coupe de fin à partir de `KEY_DURATION` (même boîte `elst`,
        // mais sans l'ambiguïté delay/padding).
        val encoderDelay = if (format.containsKey(KEY_ENCODER_DELAY))
            format.getInteger(KEY_ENCODER_DELAY) else 0
        val encoderPadding = if (format.containsKey(KEY_ENCODER_PADDING))
            format.getInteger(KEY_ENCODER_PADDING) else 0
        val durationUs = if (format.containsKey(MediaFormat.KEY_DURATION))
            format.getLong(MediaFormat.KEY_DURATION) else -1L

        // On demande du PCM float — la plupart des décodeurs logiciels
        // arm64 (dont l'AAC software du SDK 35) l'honorent ; sinon on relit
        // le format de sortie réel (voir la boucle) et on retombe sur du
        // PCM 16 bits.
        format.setInteger(MediaFormat.KEY_PCM_ENCODING, AudioFormat.ENCODING_PCM_FLOAT)

        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        val pcmChunks = ArrayList<FloatArray>()
        var totalFrames = 0L
        val bufferInfo = MediaCodec.BufferInfo()
        var sawInputEOS = false
        var sawOutputEOS = false
        var outputChannels = channelCount
        var pcmEncoding = AudioFormat.ENCODING_PCM_16BIT
        var nativeSr = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)

        while (!sawOutputEOS) {
            if (!sawInputEOS) {
                val inIndex = codec.dequeueInputBuffer(10_000)
                if (inIndex >= 0) {
                    val inBuf = codec.getInputBuffer(inIndex)!!
                    val sampleSize = extractor.readSampleData(inBuf, 0)
                    if (sampleSize < 0) {
                        codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        sawInputEOS = true
                    } else {
                        val pts = extractor.sampleTime
                        codec.queueInputBuffer(inIndex, 0, sampleSize, pts, 0)
                        extractor.advance()
                    }
                }
            }

            val outIndex = codec.dequeueOutputBuffer(bufferInfo, 10_000)
            when {
                outIndex >= 0 -> {
                    if (bufferInfo.size > 0) {
                        val outBuf = codec.getOutputBuffer(outIndex)!!
                        outBuf.position(bufferInfo.offset)
                        outBuf.limit(bufferInfo.offset + bufferInfo.size)
                        val chunk = if (pcmEncoding == AudioFormat.ENCODING_PCM_FLOAT)
                            pcmFloatToMono(outBuf, outputChannels)
                        else
                            pcm16ToFloatMono(outBuf, outputChannels)
                        pcmChunks.add(chunk)
                        totalFrames += chunk.size
                    }
                    codec.releaseOutputBuffer(outIndex, false)
                    if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                        sawOutputEOS = true
                    }
                }
                outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    val outFormat = codec.outputFormat
                    outputChannels = outFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                    nativeSr = outFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                    pcmEncoding = if (outFormat.containsKey(MediaFormat.KEY_PCM_ENCODING))
                        outFormat.getInteger(MediaFormat.KEY_PCM_ENCODING)
                    else AudioFormat.ENCODING_PCM_16BIT
                }
                // INFO_TRY_AGAIN_LATER : rien à faire, on reboucle.
            }
        }

        codec.stop()
        codec.release()
        extractor.release()

        var mono = FloatArray(totalFrames.toInt())
        var pos = 0
        for (chunk in pcmChunks) {
            System.arraycopy(chunk, 0, mono, pos, chunk.size)
            pos += chunk.size
        }

        // Délai d'amorçage / padding AAC (boîte `elst`) — le Mac (CoreAudio
        // via audioread.macca, PAS ffmpeg comme le supposait p0_t3) rend
        // exactement duration_ffprobe * sr_natif échantillons sur les 3
        // morceaux de seed (mesuré, p2a) : la boîte `elst` EST honorée côté
        // Mac. Si Android ne remplit pas ces clés, `encoderDelay`/
        // `encoderPadding` restent à 0 et ce bloc ne fait rien — le rapport
        // p2a documente alors l'écart mesuré au lieu de le deviner.
        // Longueur attendue depuis KEY_DURATION (même boîte `elst` que le
        // Mac/CoreAudio honore, cf. commentaire ci-dessus) : coupe UNIQUEMENT
        // la fin — le début est déjà juste (mesuré, décalage nul sans trim
        // de départ). Repli si KEY_DURATION absent : ne rien couper (jamais
        // vu en pratique sur les 3 morceaux de seed, gardé pour ne pas
        // planter sur un fichier sans métadonnée de durée).
        val expectedFrames = if (durationUs >= 0)
            Math.round(durationUs / 1_000_000.0 * nativeSr).toInt().coerceIn(0, mono.size)
        else mono.size

        android.util.Log.i(TAG, "decodeNativeMono $path: rawFrames=${mono.size} " +
            "encoderDelay=$encoderDelay encoderPadding=$encoderPadding durationUs=$durationUs " +
            "expectedFrames=$expectedFrames nativeSr=$nativeSr channelCount=$channelCount " +
            "outputChannels=$outputChannels")

        if (expectedFrames < mono.size) {
            mono = mono.copyOfRange(0, expectedFrames)
        }
        android.util.Log.i(TAG, "decodeNativeMono $path: trimmedFrames=${mono.size}")

        return Pair(mono, nativeSr)
    }

    private fun pcm16ToFloatMono(buf: ByteBuffer, channels: Int): FloatArray {
        buf.order(ByteOrder.LITTLE_ENDIAN)
        val shortBuf = buf.asShortBuffer()
        val nFrames = shortBuf.remaining() / channels
        val out = FloatArray(nFrames)
        val inv = 1.0f / 32768.0f
        val frame = ShortArray(channels)
        for (i in 0 until nFrames) {
            shortBuf.get(frame)
            var sum = 0f
            for (c in 0 until channels) sum += frame[c] * inv
            out[i] = sum / channels   // moyenne simple — comme librosa.to_mono
        }
        return out
    }

    private fun pcmFloatToMono(buf: ByteBuffer, channels: Int): FloatArray {
        buf.order(ByteOrder.LITTLE_ENDIAN)
        val floatBuf = buf.asFloatBuffer()
        val nFrames = floatBuf.remaining() / channels
        val out = FloatArray(nFrames)
        val frame = FloatArray(channels)
        for (i in 0 until nFrames) {
            floatBuf.get(frame)
            var sum = 0f
            for (c in 0 until channels) sum += frame[c]
            out[i] = sum / channels
        }
        return out
    }

    // ──────────────── rééchantillonnage : FIR polyphase Kaiser-sinc ────────

    /**
     * Rééchantillonnage bande-limitée par sinc fenêtré de Kaiser (beta=8.6),
     * implémentation directe (pas de FFT). Prototypé et vérifié en numpy
     * contre `librosa.resample(res_type="soxr_hq")` AVANT d'être porté ici
     * (CLAUDE.md règle #2 — écran de la prémisse) : diff nulle sur les 2000
     * premiers échantillons, corrélation croisée à décalage 0 sur une
     * fenêtre de 10 s (voir p2a pour les chiffres exacts par morceau).
     */
    fun resample(input: FloatArray, srIn: Int, srOut: Int): FloatArray {
        if (srIn == srOut) return input
        val g = gcd(srIn, srOut)
        val up = srOut / g
        val down = srIn / g
        val tapsPerPhase = 64
        val cutoff = 0.5 / maxOf(up, down)
        val filter = designKaiserSincFilter(up, tapsPerPhase, cutoff)
        return polyphaseResample(input, up, down, filter)
    }

    private fun gcd(a: Int, b: Int): Int {
        var x = a; var y = b
        while (y != 0) { val t = y; y = x % y; x = t }
        return x
    }

    private fun besselI0(x: Double): Double {
        var sum = 1.0
        var term = 1.0
        var k = 1
        while (term > sum * 1e-12) {
            term *= (x / (2.0 * k)) * (x / (2.0 * k))
            sum += term
            k++
        }
        return sum
    }

    /** Filtre prototype sinc*Kaiser, longueur 2*tapsPerPhase*up+1, gain unité en DC. */
    private fun designKaiserSincFilter(up: Int, tapsPerPhase: Int, cutoff: Double): DoubleArray {
        val halfLen = tapsPerPhase * up
        val len = 2 * halfLen + 1
        // beta=12.26 = cible d'atténuation 120 dB (formule Kaiser standard,
        // beta=0.1102*(A-8.7)), au lieu de 8.6 — mesuré en numpy (P2a-bis,
        // docs/portage_mobile/p2a_bis_resampler.md) : -4 à -8% d'écart max vs
        // soxr_hq sur les 5 cas testés, à coût de calcul IDENTIQUE (beta ne
        // change pas le nombre de coefficients). tapsPerPhase (64→512) et un
        // cutoff resserré (<100%) ont été essayés et n'ont PAS aidé — voir le
        // rapport : l'écart résiduel (jusqu'à 0.08-0.14) est concentré sur
        // des transitoires forts (Gibbs/réponse transitoire, pas une bande
        // passante trop étroite), donc insensible à la longueur du filtre.
        val beta = 12.26
        val i0Beta = besselI0(beta)
        val h = DoubleArray(len)
        for (n in 0 until len) {
            val m = n - halfLen
            val sinc = if (m == 0) 2.0 * cutoff else sin(2.0 * PI * cutoff * m) / (PI * m)
            val r = m.toDouble() / halfLen
            val w = if (abs(r) <= 1.0) besselI0(beta * sqrt(1.0 - r * r)) / i0Beta else 0.0
            h[n] = sinc * w
        }
        var sum = 0.0
        for (v in h) sum += v
        val scale = up / sum   // gain unité par phase après upsampling par `up`
        for (n in h.indices) h[n] = h[n] * scale
        return h
    }

    /**
     * y[n] = sum_i h[m - up*i] * x[i], m = n*down + halfLen (compensation du
     * délai du filtre, centré), pour les i où 0 <= m-up*i < len(h) — la
     * forme polyphase standard (cf. `scipy.signal.resample_poly`), sans
     * jamais matérialiser les zéros de l'upsampling.
     */
    private fun polyphaseResample(x: FloatArray, up: Int, down: Int, h: DoubleArray): FloatArray {
        val halfLen = (h.size - 1) / 2
        val nOut = ((x.size.toLong() * up) / down).toInt()
        val out = FloatArray(nOut)
        for (n in 0 until nOut) {
            val m = n.toLong() * down + halfLen
            var k = (m % up).toInt()
            if (k < 0) k += up
            var i = (m - k) / up
            var f = k
            var acc = 0.0
            while (f < h.size) {
                if (i >= 0 && i < x.size) {
                    acc += h[f] * x[i.toInt()]
                }
                i -= 1
                f += up
            }
            out[n] = acc.toFloat()
        }
        return out
    }
}
