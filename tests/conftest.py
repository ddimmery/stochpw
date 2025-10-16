"""Pytest configuration for stochpw tests.

This configuration allows disabling JAX JIT compilation for faster test development.
By default, JIT is ENABLED (production behavior).

To disable JIT for faster iteration during development:
    pytest --no-jit

This is useful when:
- Making quick code changes and want to skip ~2s JIT compilation overhead per test file
- Debugging tests (easier to step through without JIT)
- Note: Disabling JIT makes the actual computation slower, but skips compilation

Markers:
    - @pytest.mark.jit: Marks tests that explicitly test JIT compilation behavior
      (these will be skipped when --no-jit is used)
"""

import pytest
from jax import config


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--no-jit",
        action="store_true",
        default=False,
        help="Disable JAX JIT compilation during tests (default: JIT enabled)",
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "jit: mark test as explicitly testing JAX JIT compilation (skipped with --no-jit)",
    )


@pytest.fixture(scope="session", autouse=True)
def configure_jax(request):
    """
    Configure JAX for testing.

    By default, JIT is ENABLED (production behavior).
    Use --no-jit flag to disable JIT compilation for faster test iteration.
    """
    no_jit = request.config.getoption("--no-jit")

    if no_jit:
        # Disable JIT compilation for faster test iteration
        config.update("jax_disable_jit", True)
        print("\n[JAX Config] JIT compilation DISABLED (use without --no-jit to enable)")
    else:
        print("\n[JAX Config] JIT compilation ENABLED (default, use --no-jit to disable)")

    yield

    # Restore default after tests
    if no_jit:
        config.update("jax_disable_jit", False)


def pytest_collection_modifyitems(config, items):
    """
    Skip tests marked with @pytest.mark.jit when --no-jit is used.

    These tests explicitly test JIT compilation behavior and won't work with JIT disabled.
    """
    no_jit = config.getoption("--no-jit")

    if no_jit:
        skip_jit = pytest.mark.skip(reason="JIT-specific test skipped (--no-jit enabled)")
        for item in items:
            if "jit" in item.keywords:
                item.add_marker(skip_jit)
