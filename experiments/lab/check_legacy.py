import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab
lab = Lab(verbose=False)
head = lab.fit_legacy(lab.train_idx)
for name, idx in (("train", lab.train_idx), ("dev", lab.dev_idx)):
    mine = lab.predict_legacy(head, idx)
    ship = lab.legacy[idx]
    print(f"{name}: max|mine-shipped|={np.abs(mine-ship).max():.6f} "
          f"corr={np.mean([np.corrcoef(mine[:,k],ship[:,k])[0,1] for k in range(6)]):.6f}")
# in-sample vs out-of-sample quality of the legacy features
fold = np.random.default_rng(123).integers(0, 5, size=len(lab.train_idx))
oof = np.zeros((len(lab.train_idx), 6))
for f in range(5):
    h = lab.fit_legacy(lab.train_idx[fold != f])
    oof[fold == f] = lab.predict_legacy(h, lab.train_idx[fold == f])
tg = lab.targets
print("\nlegacy score-column corr with truth:")
for k, nm in enumerate(("light", "mid", "k1")):
    a = np.corrcoef(lab.legacy[lab.train_idx][:, k], tg[lab.train_idx][:, k])[0, 1]
    b = np.corrcoef(oof[:, k], tg[lab.train_idx][:, k])[0, 1]
    c = np.corrcoef(lab.legacy[lab.dev_idx][:, k], tg[lab.dev_idx][:, k])[0, 1]
    print(f"  {nm:6s} train in-sample={a:.3f}  train OOF={b:.3f}  dev out-of-sample={c:.3f}")
print("legacy log-cost RMSE:")
for k, nm in enumerate(("light", "mid", "k1")):
    a = np.sqrt(((lab.legacy[lab.train_idx][:,3+k]-tg[lab.train_idx][:,3+k])**2).mean())
    b = np.sqrt(((oof[:,3+k]-tg[lab.train_idx][:,3+k])**2).mean())
    c = np.sqrt(((lab.legacy[lab.dev_idx][:,3+k]-tg[lab.dev_idx][:,3+k])**2).mean())
    print(f"  {nm:6s} train in-sample={a:.3f}  train OOF={b:.3f}  dev={c:.3f}")
