"""H54 exploration: which regions should get a background inverse? (ADMIT = all | big | local)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_real, rsolve
from btfd import BTFD
for case in sys.argv[2:]:
    A, b = run_real.load(case)
    for admit in sys.argv[1].split(","):
        rsolve.MultiFDLGF.ADMIT = admit
        P = BTFD(A, multi=True)
        out = []
        for tol in (1e-4, 1e-8):
            r = run_real.run(A, b, "bt-fdm:2", tol)
            out.append(f"{tol:g}: {r['its']} its {r['total_s']:.1f} s (setup {r['setup_s']:.1f})")
        print(f"{case} admit {admit:5s} feature S3 {len(P.p3.F) / P.N:.2%} A11 {len(P.p1.F) / P.N:.2%} | " + " | ".join(out), flush=True)
