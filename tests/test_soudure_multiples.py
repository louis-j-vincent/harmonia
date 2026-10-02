"""Une occurrence tracée sur k× la longueur habituelle de sa lettre = k passages
de cette lettre, pas une variante A′ (Louis, 2026-10-02, Go Thru Your Phone :
« le A et le A′ sont équivalents, il faudrait les mettre ensemble » — A tracé
partout sur 2 mesures, une fois sur 4 → l'écriture en faisait un A′ que
aucun cran du curseur de fusion ne pouvait rejoindre). Même règle que la
boucle minimale de la recherche de reprises (2026-09-30)."""
from harmonia.soudure import scinder_multiples


def S(label, a, b):
    return {"label": label, "mesure_debut": a, "mesure_fin": b}


def test_double_length_is_two_passes():
    secs = [S("intro", 1, 3), S("A", 4, 5), S("A", 6, 7), S("B", 8, 11),
            S("A", 12, 15), S("outro", 16, 20)]
    out = scinder_multiples(secs)
    assert [(s["label"], s["mesure_debut"], s["mesure_fin"]) for s in out] == [
        ("intro", 1, 3), ("A", 4, 5), ("A", 6, 7), ("B", 8, 11),
        ("A", 12, 13), ("A", 14, 15), ("outro", 16, 20)]


def test_non_multiple_and_other_labels_untouched():
    secs = [S("A", 1, 4), S("A", 5, 8), S("A", 9, 14), S("A′", 15, 22),
            S("intro", 23, 30), S("intro", 31, 34)]
    out = scinder_multiples(secs)
    assert out == secs                     # 6 n'est pas un multiple de 4 ; A′ et intro ne sont pas des lettres


def test_single_occurrence_letter_untouched():
    secs = [S("A", 1, 8), S("B", 9, 12)]
    assert scinder_multiples(secs) == secs
