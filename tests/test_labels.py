from isl_recognition.labels import ALL_LABELS, DIGIT_LABELS, STATIC_LABELS


def test_static_labels_cover_letters_and_digits():
    assert len(STATIC_LABELS) == 36
    assert STATIC_LABELS == ALL_LABELS + DIGIT_LABELS
