from shutil import copyfile

from PIL import Image

from isl_recognition.data.validate import inspect_dataset


def test_validation_counts_only_readable_unambiguous_samples(tmp_path):
    image_root = tmp_path / "original_images"
    class_a = image_root / "A"
    class_b = image_root / "B"
    class_a.mkdir(parents=True)
    class_b.mkdir()

    image_path = class_a / "sample.png"
    Image.new("RGB", (8, 8), (20, 40, 60)).save(image_path)
    copyfile(image_path, class_a / "duplicate.png")
    copyfile(image_path, class_b / "conflicting_duplicate.png")
    (class_a / "broken.png").write_text("not an image", encoding="utf-8")

    report = inspect_dataset(tmp_path)

    assert report["class_counts"]["A"] == 2
    assert report["class_counts"]["B"] == 0
    assert report["exact_duplicate_count"] == 1
    assert len(report["exact_duplicate_examples"]) == 1
    assert len(report["invalid_or_conflicting_samples"]) == 2
    assert report["status"] == "needs_attention"
