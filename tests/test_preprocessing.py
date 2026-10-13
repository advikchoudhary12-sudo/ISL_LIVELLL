import numpy as np
from PIL import Image

from isl_recognition.preprocessing import preprocess_image, preprocess_path, preprocessing_config


def test_preprocessing_is_consistent_for_image_and_path(tmp_path):
    image = Image.new("RGB", (120, 60), color=(15, 90, 210))
    path = tmp_path / "sample.png"
    image.save(path)

    from_image = preprocess_image(image, size=96)
    from_path = preprocess_path(path, size=96)

    assert from_image.shape == (3, 96, 96)
    assert from_image.dtype == np.float32
    assert np.array_equal(from_image, from_path)
    assert 0.0 <= from_image.min() <= from_image.max() <= 1.0


def test_preprocessing_rejects_invalid_size():
    try:
        preprocess_image(Image.new("RGB", (10, 10)), size=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("size=0 must be rejected")


def test_preprocessing_configuration_records_static_image_inputs():
    static = preprocessing_config(96)

    assert static["frame"]["value_range"] == [0.0, 1.0]
    assert static["frame"]["tensor_layout"] == "CHW"
