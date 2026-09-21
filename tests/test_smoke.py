from src.label_loader import map_sleep_stage_to_binary


def test_map_sleep_stage_to_binary():
    import pandas as pd

    labels = pd.DataFrame({"time": [0, 30, 60, 90, 120], "stage": [0, 1, 2, 3, 5]})
    mapped = map_sleep_stage_to_binary(labels)
    assert mapped["label"].tolist() == [0, 1, 1, 1, 1]
