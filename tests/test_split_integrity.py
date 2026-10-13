from isl_recognition.data.prepare import _split_groups
from isl_recognition.labels import STATIC_LABELS


def test_static_visual_duplicate_groups_never_cross_splits():
    records = [
        {
            "label": label,
            "path": f"{label}/{group}/{item}.jpg",
            "group_id": f"{label}-{group}",
        }
        for label in STATIC_LABELS
        for group in range(10)
        for item in range(2)
    ]
    splits = _split_groups(records, seed=42)
    group_sets = {
        split: {row["group_id"] for row in rows}
        for split, rows in splits.items()
    }
    assert group_sets["train"].isdisjoint(group_sets["validation"])
    assert group_sets["train"].isdisjoint(group_sets["test"])
    assert group_sets["validation"].isdisjoint(group_sets["test"])
    for rows in splits.values():
        assert {row["label"] for row in rows} == set(STATIC_LABELS)
