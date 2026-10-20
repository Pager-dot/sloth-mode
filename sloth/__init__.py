"""Sloth Mode — local, open-weight AI that does the boring work while you nap.

Nothing in here talks to a cloud API. Everything runs against a model you
host yourself (Ollama by default, or any OpenAI-compatible local server).
The only dependency is the Python standard library, because installing
things is work.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
