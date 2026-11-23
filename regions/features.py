"""H52 prediction 5: feature fraction (before the ring) of both scalar blocks on the 8 test matrices."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_real
from btfd import BTFD
for case in ["real"] + [f"geo:128:{c}" for c in ("43:54:1:2e-11:2e-4", "25:32:1:2e-11:2e-4", "60:66:1:2e-11:2e-4",
                                                   "30:70:1:2e-11:2e-4", "43:54:0.1:2e-11:2e-4", "43:54:10:2e-11:2e-4",
                                                   "43:54:1:1e-3:1e-2")]:
    A, b = run_real.load(case)
    P = BTFD(A, ring=2)
    print(f"{case}: S3 feature {P.p3.core_fraction:.2%} (with ring {len(P.p3.F) / P.N:.2%}), "
          f"A11 feature {P.p1.core_fraction:.2%} (with ring {len(P.p1.F) / P.N:.2%})", flush=True)
