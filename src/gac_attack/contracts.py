"""Frozen clean-input contracts used by the public runner."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping


_VALID_ROLES = {"answer", "auxiliary", "context"}


@dataclass(frozen=True)
class Page:
    path: str
    sha256: str
    role: str


@dataclass(frozen=True)
class InputContract:
    sample_id: str
    prompt: str
    answer_aliases: tuple[str, ...]
    pages: tuple[Page, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InputContract":
        try:
            raw_pages = value["pages"]
            if not isinstance(raw_pages, (list, tuple)):
                raise ValueError("pages must be a list")
            pages = tuple(
                Page(path=str(page["path"]), sha256=str(page["sha256"]), role=str(page["role"]))
                for page in raw_pages
            )
            contract = cls(
                sample_id=str(value["sample_id"]),
                prompt=str(value["prompt"]),
                answer_aliases=tuple(str(alias) for alias in value["answer_aliases"]),
                pages=pages,
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"MISSING_CONTRACT_FIELD:{exc.args[0]}") from exc
        contract.validate()
        return contract

    def validate(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        if not self.answer_aliases:
            raise ValueError("answer_aliases must not be empty")
        if not self.pages:
            raise ValueError("pages must not be empty")
        if not any(page.role == "auxiliary" for page in self.pages):
            raise ValueError("contract requires at least one auxiliary page")
        for page in self.pages:
            if page.role not in _VALID_ROLES:
                raise ValueError(f"unknown page role: {page.role}")
            if len(page.sha256) != 64 or any(char not in "0123456789abcdef" for char in page.sha256.lower()):
                raise ValueError(f"invalid sha256 for {page.path}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_contract_files(contract: InputContract, root: Path) -> list[str]:
    """Return deterministic errors; an empty list means all pages match."""
    errors: list[str] = []
    root = root.resolve()
    for page in contract.pages:
        path = (root / page.path).resolve(strict=False)
        if not path.is_relative_to(root):
            errors.append(f"PATH_OUTSIDE_ROOT:{page.path}")
            continue
        if not path.is_file():
            errors.append(f"MISSING_PAGE:{page.path}")
        elif sha256_file(path) != page.sha256.lower():
            errors.append(f"SHA256_MISMATCH:{page.path}")
    return errors
