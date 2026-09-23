"""Run the original crossed-grid code with a separate reproduction output path."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'arc_lesion_image_benchmark'
sys.path.insert(0, str(PKG / 'src'))
import s18_crossed_grid as grid

if __name__ == '__main__':
    # TABLE and SUBSET_DIR keep pointing to the archived inputs; only the
    # original main() output root is redirected.
    grid.PKG = str(PKG / 'reproduced/grid')
    grid.main()
