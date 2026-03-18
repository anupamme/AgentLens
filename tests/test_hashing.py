"""Tests for the hashing utilities."""

from agentlens.utils.hashing import hash_content, hash_input


class TestHashInput:
    def test_determinism(self):
        h1 = hash_input("hello world")
        h2 = hash_input("hello world")
        assert h1 == h2

    def test_collision_resistance(self):
        h1 = hash_input("input_a")
        h2 = hash_input("input_b")
        assert h1 != h2

    def test_xxhash_format(self):
        result = hash_input("test data")
        assert result.startswith("xxh64:")
        hex_part = result.split(":")[1]
        assert len(hex_part) == 16  # xxh64 produces 16 hex chars
        assert all(c in "0123456789abcdef" for c in hex_part)

    def test_sha256_format(self):
        result = hash_input("test data", method="sha256")
        assert result.startswith("sha256:")
        hex_part = result.split(":")[1]
        assert len(hex_part) == 64  # sha256 produces 64 hex chars


class TestHashContent:
    def test_determinism(self):
        h1 = hash_content("some content")
        h2 = hash_content("some content")
        assert h1 == h2

    def test_format(self):
        result = hash_content("test")
        assert result.startswith("xxh64:")
