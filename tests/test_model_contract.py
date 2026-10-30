from __future__ import annotations

import copy

import pytest

from model_contract import (
    CLASSIFIER_FOLDER_TO_INDEX,
    CLASSIFIER_LABELS,
    GENERATOR_CONDITIONS,
    ContractError,
    classifier_metadata,
    generator_metadata,
    validate_classifier_metadata,
)


def test_classifier_labels_and_folder_mapping() -> None:
    assert CLASSIFIER_LABELS == ("cat", "not_cat")
    assert CLASSIFIER_FOLDER_TO_INDEX == {"cat": 0, "other": 1}
    metadata = classifier_metadata()
    assert len(CLASSIFIER_LABELS) == metadata["architecture"]["num_classes"]


def test_generator_conditions_and_unconditional_index() -> None:
    expected = (
        "Abyssinian",
        "Bengal",
        "Birman",
        "Bombay",
        "British_Shorthair",
        "Egyptian_Mau",
        "Maine_Coon",
        "Persian",
        "Ragdoll",
        "Russian_Blue",
        "Siamese",
        "Sphynx",
        "Other",
    )
    assert expected == GENERATOR_CONDITIONS
    metadata = generator_metadata()
    assert {row["index"]: row["name"] for row in metadata["conditions"]} == dict(
        enumerate(expected)
    )
    assert metadata["unconditional_index"] == 13


def test_classifier_metadata_round_trips() -> None:
    metadata = classifier_metadata()
    assert (
        validate_classifier_metadata(
            metadata, expected_num_classes=2, expected_labels=CLASSIFIER_LABELS
        )
        == metadata
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["architecture"].update(num_classes=13),
        lambda value: value["output"].update(labels=["cat"]),
        lambda value: value["preprocessing"].update(mean=[0.0, 0.0, 0.0]),
        lambda value: value["input"].update(shape=[1, 3, 256, 256]),
    ],
    ids=["wrong-output-dim", "wrong-labels", "wrong-mean", "wrong-input-size"],
)
def test_classifier_metadata_rejects_mismatches(mutate: object) -> None:
    metadata = copy.deepcopy(classifier_metadata())
    mutate(metadata)  # type: ignore[operator]
    with pytest.raises(ContractError):
        validate_classifier_metadata(metadata, expected_num_classes=2)
