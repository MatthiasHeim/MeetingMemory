"""Tests for calendar_resolve — humanising the email-local-part fallback that
used to leak handles like 'sascha.lioi' into participant names when a calendar
attendee has no displayName."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from calendar_resolve import _humanize_email_local  # noqa: E402


def test_humanize_firstname_lastname():
    """The common external-attendee pattern firstname.lastname → 'First Last'."""
    assert _humanize_email_local("sascha.lioi") == "Sascha Lioi"
    assert _humanize_email_local("lorenz.fehr") == "Lorenz Fehr"
    assert _humanize_email_local("phillip.fumolo") == "Phillip Fumolo"
    assert _humanize_email_local("philipp.baltensperger") == "Philipp Baltensperger"


def test_humanize_underscore_and_hyphen_separators():
    assert _humanize_email_local("max_mustermann") == "Max Mustermann"
    assert _humanize_email_local("anna-lena.huber") == "Anna Lena Huber"


def test_humanize_leaves_non_name_locals_untouched():
    """Role/system addresses and single tokens must NOT be turned into fake
    names — better to keep the raw local than invent 'Info' / 'Noreply'."""
    for local in ("info", "noreply", "mheim", "support"):
        assert _humanize_email_local(local) == local


def test_humanize_leaves_locals_with_digits_untouched():
    """Anything with a non-alpha token is left as-is (not name-shaped)."""
    assert _humanize_email_local("sascha.lioi2") == "sascha.lioi2"
    assert _humanize_email_local("user.123") == "user.123"
