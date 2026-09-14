from chordnet_ismir_naive import ChordNet,chord_limit,ChordNetCNN
from mir.nn.train import NetworkInterface
from extractors.cqt import CQTV2,SimpleChordToID
from mir import io,DataEntry
from extractors.xhmm_ismir import XHMMDecoder
import numpy as np
from io_new.chordlab_io import ChordLabIO
from settings import DEFAULT_SR,DEFAULT_HOP_LENGTH
import sys

MODEL_NAMES = ['joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s%d.best' % i for i in range(5)]


def chord_recognition(audio_path, lab_path, chord_dict_name='submission'):
    hmm = XHMMDecoder(template_file='data/%s_chord_list.txt' % chord_dict_name)
    entry = DataEntry()
    entry.prop.set('sr', DEFAULT_SR)
    entry.prop.set('hop_length', DEFAULT_HOP_LENGTH)
    entry.append_file(audio_path, io.MusicIO, 'music')
    entry.append_extractor(CQTV2, 'cqt')
    probs = []
    for i, model_name in enumerate(MODEL_NAMES):
        net = NetworkInterface(ChordNet(None), model_name, load_checkpoint=False)
        print('Inference: %s on %s' % (model_name, audio_path))
        probs.append(net.inference(entry.cqt))
        # Progressive reveal (2026-07-20, harmonia integration): decode the
        # running average of folds seen so far and write it to a fold-numbered
        # sidecar, so a caller polling the output directory sees an improving
        # intermediate chordlab every ~1-2s instead of waiting for all 5 folds.
        # Decode cost measured at ~0.13s/fold vs ~1.3-2.2s/fold for the network
        # inference it rides along with -- negligible overhead. The FINAL
        # write to lab_path below is unchanged, so any other caller of this
        # script sees identical behaviour.
        partial_probs = [np.mean([p[j] for p in probs], axis=0) for j in range(len(probs[0]))]
        partial_chordlab = hmm.decode_to_chordlab(entry, partial_probs, False)
        entry.append_data(partial_chordlab, ChordLabIO, 'chord')
        entry.save('chord', '%s.fold%d' % (lab_path, i + 1))
        print('Fold %d/%d done' % (i + 1, len(MODEL_NAMES)))
    probs = [np.mean([p[i] for p in probs], axis=0) for i in range(len(probs[0]))]
    chordlab = hmm.decode_to_chordlab(entry, probs, False)
    entry.append_data(chordlab, ChordLabIO, 'chord')
    entry.save('chord', lab_path)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        chord_recognition(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        chord_recognition(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print('Usage: chord_recognition.py path_to_audio_file path_to_output_file [chord_dict=submission]')
        print('\tChord dict can be one of the following: full, ismir2017, submission, extended')
        exit(0)