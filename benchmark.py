#!/usr/bin/env python
"""Query, reproduce and compare against the ARC reference benchmark."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "arc_lesion_image_benchmark/src"))
from arc_reference import main

if __name__ == "__main__":
    main()
