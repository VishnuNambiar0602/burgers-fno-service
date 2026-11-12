"""Test defaults: the rate limit is OFF (the tests make many requests) and MCP is ON (the MCP tests need it).
Tests that check the real defaults delete these variables themselves (tests/test_limits.py)."""
import os

os.environ.setdefault("RATE_LIMIT_PER_MIN", "0")
os.environ.setdefault("MCP_ENABLED", "1")
