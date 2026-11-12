"""Helpers shared by the backend tests."""
import urllib.request

import pytest


@pytest.fixture
def failing_urlopen(monkeypatch):
    """fail(*errors): urlopen raises these errors in turn, then works again."""
    def fail(*errors):
        real, left = urllib.request.urlopen, list(errors)

        def urlopen(req, timeout=None):
            if left:
                raise left.pop(0)
            return real(req, timeout=timeout)
        monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return fail
