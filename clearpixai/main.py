"""Entry point for ClearPixAi CLI."""

from .cli import main

__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
