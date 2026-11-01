"""Who a memory was written for: the host's viewer behind an embed or a service write."""
from __future__ import annotations

import re
from typing import Any

from papez.models.ref import ValidationFailure

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_ID_MAX = 128
_EMAIL_MAX = 254
_NAME_MAX = 120


def validate_viewer(value: Any) -> dict[str, Any]:
    """Return ``{"id", "email", "name"}`` for a viewer object; unknown keys are dropped.

    ``id`` is required (1-128 characters). ``email`` is optional and must look
    like an address (at most 254 characters). ``name`` is optional (at most 120
    characters). Anything else raises ``ValidationFailure("invalid_viewer", ...)``.
    """
    if not isinstance(value, dict):
        raise ValidationFailure("invalid_viewer", "viewer must be an object")
    viewer_id = value.get("id")
    if not isinstance(viewer_id, str) or not 1 <= len(viewer_id) <= _ID_MAX:
        raise ValidationFailure("invalid_viewer", f"viewer.id must be a string of 1-{_ID_MAX} characters")
    email = value.get("email")
    if email is not None and (
        not isinstance(email, str) or len(email) > _EMAIL_MAX or not _EMAIL_RE.match(email)
    ):
        raise ValidationFailure("invalid_viewer", "viewer.email must be a valid email address")
    name = value.get("name")
    if name is not None and (not isinstance(name, str) or len(name) > _NAME_MAX):
        raise ValidationFailure("invalid_viewer", f"viewer.name must be a string of at most {_NAME_MAX} characters")
    return {"id": viewer_id, "email": email, "name": name}
