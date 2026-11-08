import hashlib
import json

import pytest

from sequence_slot_attack.contracts import InputContract, verify_contract_files


def test_contract_verifies_named_clean_page_hash(tmp_path):
    page = tmp_path / "page.png"
    page.write_bytes(b"clean-page")
    contract = InputContract.from_mapping(
        {
            "sample_id": "example-001",
            "prompt": "Read the value.",
            "answer_aliases": ["42"],
            "pages": [
                {
                    "path": "page.png",
                    "sha256": hashlib.sha256(b"clean-page").hexdigest(),
                    "role": "auxiliary",
                }
            ],
        }
    )

    assert verify_contract_files(contract, tmp_path) == []


def test_contract_reports_changed_page_hash(tmp_path):
    page = tmp_path / "page.png"
    page.write_bytes(b"changed")
    contract = InputContract.from_mapping(
        {
            "sample_id": "example-001",
            "prompt": "Read the value.",
            "answer_aliases": ["42"],
            "pages": [{"path": "page.png", "sha256": "0" * 64, "role": "auxiliary"}],
        }
    )

    assert verify_contract_files(contract, tmp_path) == ["SHA256_MISMATCH:page.png"]


def test_contract_requires_an_auxiliary_page():
    with pytest.raises(ValueError, match="auxiliary"):
        InputContract.from_mapping(
            {
                "sample_id": "example-001",
                "prompt": "Read the value.",
                "answer_aliases": ["42"],
                "pages": [{"path": "page.png", "sha256": "0" * 64, "role": "answer"}],
            }
        )


def test_contract_rejects_a_non_list_pages_field():
    with pytest.raises(ValueError, match="pages"):
        InputContract.from_mapping(
            {"sample_id": "example", "prompt": "p", "answer_aliases": ["a"], "pages": "not-a-list"}
        )


def test_contract_file_verification_rejects_paths_outside_root(tmp_path):
    contract = InputContract.from_mapping(
        {
            "sample_id": "example",
            "prompt": "p",
            "answer_aliases": ["a"],
            "pages": [{"path": "../outside.png", "sha256": "0" * 64, "role": "auxiliary"}],
        }
    )
    assert verify_contract_files(contract, tmp_path) == ["PATH_OUTSIDE_ROOT:../outside.png"]
