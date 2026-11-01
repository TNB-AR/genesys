"""References (provenance pointers), tag normalization and ref validation (spec 5.2, 5.3).

The limits are passed in explicitly (D34): callers read ``config.MAX_TAG_LENGTH``
etc. at call time, so ``models/`` keeps importing nothing from ``engine/``.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

REF_TYPE_PATTERN = r"^[a-z0-9][a-z0-9_.-]{0,63}$"   # e.g. "document", "meeting", "crm.contact"

_TAG_ALLOWED = re.compile(r"^[a-z0-9_.:-]+$")
_WHITESPACE_RUN = re.compile(r"\s+")


class Ref(BaseModel):
    """A pointer from a memory to the thing it came from (a document, a meeting, a ticket...)."""

    type: str = Field(pattern=REF_TYPE_PATTERN)          # what kind of thing: "document", "meeting", "url", ...
    id: str = Field(min_length=1, max_length=256)        # the id in the source system
    url: str | None = Field(default=None, max_length=2048)
    label: str | None = Field(default=None, max_length=200)   # human-readable name, e.g. a file name

    @field_validator("url")
    @classmethod
    def url_must_be_http(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith(("https://", "http://")):
            raise ValueError("url must start with http:// or https://")
        return v


class ValidationFailure(ValueError):
    """A tag/ref validation failure carrying a spec Section 10 error code.

    ``code`` is one of ``invalid_tag``, ``too_many_tags``, ``invalid_ref``,
    ``too_many_refs``; ``message`` is the human-readable text. ``str(exc)`` is
    the message. ``as_error()`` returns the tool error dict ``{"error", "code"}``.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def as_error(self) -> dict[str, str]:
        return {"error": self.message, "code": self.code}


def normalize_tags(tags: Iterable[Any] | None, *, max_tag_length: int, max_tags: int) -> list[str]:
    """Normalize tags per spec 5.3; raise ``ValidationFailure`` on any violation.

    1. Trim whitespace and lowercase.
    2. Replace runs of whitespace with a single ``-``.
    3. Allowed characters: ``a-z 0-9 - _ . :``; anything else -> ``invalid_tag``.
    4. Drop empty tags.
    5. Remove duplicates, keeping the first occurrence's order.
    6. A tag longer than ``max_tag_length`` -> ``invalid_tag``; more than
       ``max_tags`` tags after steps 1-5 -> ``too_many_tags``.

    ``None`` means no tags and returns ``[]``.
    """
    if tags is None:
        return []
    if isinstance(tags, (str, bytes)):
        raise ValidationFailure("invalid_tag", "tags must be a list of strings")
    out: list[str] = []
    seen: set[str] = set()
    for raw in tags:
        if not isinstance(raw, str):
            raise ValidationFailure("invalid_tag", f"tag {raw!r} must be a string")
        tag = _WHITESPACE_RUN.sub("-", raw.strip().lower())
        if not tag:
            continue
        if not _TAG_ALLOWED.match(tag):
            raise ValidationFailure("invalid_tag", f"tag '{raw.strip()}' contains invalid characters")
        if len(tag) > max_tag_length:
            raise ValidationFailure(
                "invalid_tag", f"tag '{tag}' is longer than {max_tag_length} characters",
            )
        if tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    if len(out) > max_tags:
        raise ValidationFailure("too_many_tags", f"{len(out)} tags given; at most {max_tags} are allowed")
    return out


def validate_refs(refs: Iterable[Any] | None, *, max_refs: int) -> list[Ref]:
    """Validate refs with the ``Ref`` model, dedupe by ``(type, id)`` keeping the first.

    Each item may be a ``Ref`` or a dict. A failing item -> ``invalid_ref``;
    more than ``max_refs`` distinct refs -> ``too_many_refs``. ``None`` returns ``[]``.
    """
    if refs is None:
        return []
    if isinstance(refs, (str, bytes, dict)):
        raise ValidationFailure("invalid_ref", "refs must be a list of objects")
    out: list[Ref] = []
    seen: set[tuple[str, str]] = set()
    for i, raw in enumerate(refs):
        if isinstance(raw, Ref):
            ref = Ref.model_validate(raw.model_dump())
        elif isinstance(raw, dict):
            try:
                ref = Ref.model_validate(raw)
            except ValidationError as exc:
                problems = "; ".join(
                    f"{'.'.join(str(p) for p in err['loc']) or 'ref'}: {err['msg']}" for err in exc.errors()
                )
                raise ValidationFailure("invalid_ref", f"ref {i} is invalid: {problems}") from None
        else:
            raise ValidationFailure("invalid_ref", f"ref {i} must be an object")
        key = (ref.type, ref.id)
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    if len(out) > max_refs:
        raise ValidationFailure("too_many_refs", f"{len(out)} refs given; at most {max_refs} are allowed")
    return out
