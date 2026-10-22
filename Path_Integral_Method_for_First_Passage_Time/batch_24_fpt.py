import itertools
from pathlib import Path
import fpt_example1 as f

out = Path(__file__).resolve().parent / "fpt_24组参数结果"
out.mkdir(exist_ok=True)
f.__file__ = str(out / "fpt_example1.py")

params = itertools.product(
    [0.6, 1.0, 1.4],  # MU
    [0.4, 0.8],       # SIGMA
    [0.0, 0.2],       # X0
    [1.0, 1.4]        # BETA
)

for i, (mu, sigma, x0, beta) in enumerate(params, 1):
    f.MU, f.SIGMA, f.X0, f.BETA = mu, sigma, x0, beta
    f.run()
    (out / "fpt_results_E1").rename(
        out / f"{i:02d}_mu{mu}_sigma{sigma}_x{x0}_beta{beta}"
    )
