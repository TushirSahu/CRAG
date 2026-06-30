"""Daily embedding-drift check (invoked by .github/workflows/monitor.yaml).

Compares the distribution of the knowledge base's embedding norms against a
saved baseline using PSI. The first run writes the baseline; later runs report
how far the corpus has drifted. Reads vectors straight from LanceDB so it needs
no embedding model, and exits cleanly when there's nothing to compare — safe to
run in CI against an empty store.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Allow ``python src/monitoring/drift_detection.py`` (CI) to resolve ``src.*``.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core import lancedb as ldb  # noqa: E402
from src.core.config import get_settings  # noqa: E402
from src.monitoring.metrics import embedding_norms, population_stability_index  # noqa: E402
from src.utils.logger import logger  # noqa: E402

REFERENCE_PATH = "data/drift_reference.json"
DRIFT_THRESHOLD = 0.25  # PSI above this is a significant shift


def _current_norms() -> np.ndarray:
    cfg = get_settings().vector_store
    db = ldb.connect(cfg.path)
    if cfg.table not in db.table_names():
        return np.array([])
    rows = db.open_table(cfg.table).to_arrow().to_pylist()
    return embedding_norms([r["vector"] for r in rows if r.get("vector") is not None])


def main() -> int:
    current = _current_norms()
    if current.size == 0:
        logger.info("Drift check: knowledge base empty, nothing to compare.")
        print("No data to monitor — skipping drift check.")
        return 0

    ref_path = Path(get_settings().resolve(REFERENCE_PATH))
    if not ref_path.exists():
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(json.dumps({"norms": current.tolist()}))
        logger.info("Drift check: saved initial baseline (%d vectors).", current.size)
        print(f"Saved drift baseline ({current.size} vectors).")
        return 0

    reference = np.array(json.loads(ref_path.read_text())["norms"])
    psi = population_stability_index(reference, current)
    status = "OK" if psi < DRIFT_THRESHOLD else "DRIFT"
    logger.info("Drift check PSI=%.4f (%s)", psi, status)
    print(f"Embedding drift PSI={psi:.4f} → {status} (threshold {DRIFT_THRESHOLD}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
