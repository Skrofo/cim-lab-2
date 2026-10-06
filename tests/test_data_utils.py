from cim_lab_2.data import breed_names, stratified_indices


def test_breed_names_follow_class_indices():
    records = [
        {"name": "american_bulldog_3", "breed": 0},
        {"name": "abyssinian_12", "breed": 1},
    ]
    assert breed_names(records, num_classes=2) == ["American Bulldog", "Abyssinian"]


def test_breed_names_reject_conflicting_class_names():
    records = [
        {"name": "american_bulldog_3", "breed": 0},
        {"name": "bengal_12", "breed": 0},
    ]
    try:
        breed_names(records, num_classes=1)
    except ValueError as error:
        assert "Conflicting names" in str(error)
    else:
        raise AssertionError("Expected conflicting breed names to raise ValueError")


def test_stratified_split_contains_every_class_in_validation():
    records = []
    for breed in range(3):
        records.extend({"breed": breed} for _ in range(10))
    train, val = stratified_indices(records, 0.2, 42)
    assert len(train) == 24
    assert len(val) == 6
    assert {records[i]["breed"] for i in val} == {0, 1, 2}
