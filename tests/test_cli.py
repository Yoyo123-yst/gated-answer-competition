import hashlib
import json

from sequence_slot_attack.cli import main


def test_validate_contract_cli_accepts_matching_file(tmp_path, capsys):
    page = tmp_path / "page.png"
    page.write_bytes(b"page")
    contract = {
        "sample_id": "public-example",
        "prompt": "Read the value.",
        "answer_aliases": ["42"],
        "pages": [{"path": "page.png", "sha256": hashlib.sha256(b"page").hexdigest(), "role": "auxiliary"}],
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))

    assert main(["validate-contract", "--contract", str(contract_path), "--root", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "VALID"


def test_validate_contract_cli_returns_json_for_malformed_pages(tmp_path, capsys):
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps({"sample_id": "x", "prompt": "p", "answer_aliases": ["a"], "pages": "bad"}))

    assert main(["validate-contract", "--contract", str(contract_path), "--root", str(tmp_path)]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "INVALID"
