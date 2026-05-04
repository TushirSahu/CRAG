"""Training orchestration for the CRAG domain-adapted embedding workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.generate_synthetic_data import generate_dataset
from src.utils.finetune_embeddings import finetune_embeddings


def run_training_pipeline(output_dir: str | Path = "./data") -> dict[str, Any]:
	"""Generate synthetic pairs and fine-tune the embedding model."""

	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	generate_dataset()
	finetune_embeddings()

	return {
		"training_pairs_csv": str(output_path / "training_pairs.csv"),
		"finetuned_model_path": str(output_path / "finetuned-domain-embeddings"),
		"status": "completed",
	}


def main() -> None:
	"""CLI entrypoint."""

	result = run_training_pipeline()
	print(result)


if __name__ == "__main__":
	main()
