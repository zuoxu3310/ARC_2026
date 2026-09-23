"""Recompute the original variance analysis without changing archived outputs."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'arc_lesion_image_benchmark'
sys.path.insert(0, str(PKG / 'src'))
import s19_variance_decomp as base

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=['regression', 'classification'])
    parser.add_argument('--include-tabpfn', action='store_true')
    args = parser.parse_args()
    base.SUMM = str(PKG / 'reproduced/variance')
    sys.argv = [sys.argv[0], '--task', args.task]
    if args.include_tabpfn:
        import s19b_variance_decomp_with_tabpfn as extended
        extended.SUMM = base.SUMM
        extended.main()
    else:
        base.main()
