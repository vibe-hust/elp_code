"""Allow invoking `python -m elp` directly."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
