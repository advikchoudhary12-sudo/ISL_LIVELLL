from isl_recognition.labels import ALL_LABELS, STATIC_LABELS


def test_static_labels_cover_full_alphabet():
    assert len(STATIC_LABELS) == 26
    assert STATIC_LABELS == ALL_LABELS
