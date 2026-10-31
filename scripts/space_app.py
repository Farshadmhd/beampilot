"""Entry point of the hosted demo (Hugging Face Space, ``app_file``).

The Space runs this file directly and does not install the package, so
``src/`` is put on the import path first. The app then finds the shipped
replays in ``docs/runs/`` next to it. See docs/hosting-plan.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beampilot.app import main  # noqa: E402

if __name__ == "__main__":
    main()
