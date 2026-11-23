"""H53 prediction 5: linearity defect ||P(x+y) - P(x) - P(y)|| / ||P(x) + P(y)|| of bt-fd on the test matrices."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_real
from btfd import BTFD
for case in sys.argv[1:]:
    A, b = run_real.load(case)
    P = BTFD(A, ring=2)
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal(A.shape[0]), rng.standard_normal(A.shape[0])
    print(f"{case}: linearity defect {np.linalg.norm(P(x + y) - P(x) - P(y)) / np.linalg.norm(P(x) + P(y)):.1e}, "
          f"feature S3 {P.p3.core_fraction:.2%} A11 {P.p1.core_fraction:.2%}", flush=True)
