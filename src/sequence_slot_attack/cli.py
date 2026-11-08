"""Dependency-light command line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .contracts import InputContract, verify_contract_files


def _validate_contract(arguments: argparse.Namespace) -> int:
    contract_path = Path(arguments.contract)
    try:
        contract = InputContract.from_mapping(json.loads(contract_path.read_text()))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "INVALID", "reason": str(exc)}, sort_keys=True))
        return 2
    errors = verify_contract_files(contract, Path(arguments.root))
    payload = {"sample_id": contract.sample_id, "status": "VALID" if not errors else "INVALID", "errors": errors}
    print(json.dumps(payload, sort_keys=True))
    return 0 if not errors else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sequence-Slot FastV release utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-contract", help="validate clean page hashes before an attack")
    validate.add_argument("--contract", required=True, help="path to an input-contract JSON file")
    validate.add_argument("--root", required=True, help="root directory resolving page paths")
    validate.set_defaults(handler=_validate_contract)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
