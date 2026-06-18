"""Shared pytest config + a `not integration` default filter."""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip @pytest.mark.integration unless ECA_RUN_INTEGRATION=1 is set."""
    import os

    if os.getenv("ECA_RUN_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="integration test; set ECA_RUN_INTEGRATION=1 to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
