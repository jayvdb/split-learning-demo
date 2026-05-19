"""Prefetch datasets used by the demo so the server can run offline.

Downloads ``ylecun/mnist`` (parquet) from the Hugging Face Hub via the
``hf`` CLI into ``data/external/mnist/``. The split-learning loader
(``split_learning.utils.datasets.mnist``) reads those files at runtime
instead of hitting the network on first use.

Usage:
    python scripts/fetch_data.py
    python scripts/fetch_data.py --dataset mnist
"""

import logging
import shutil
import subprocess
import sys

import click

from split_learning.utils import utils

_logger = logging.getLogger(__name__)

# Maps the demo's logical dataset name -> (HF repo id, local subdir under data/external).
_DATASETS = {
    "mnist": ("ylecun/mnist", "mnist"),
}


def _hf_cli() -> list[str]:
    """Return the argv prefix for invoking the Hugging Face Hub CLI.

    Prefers the new short ``hf`` binary (huggingface_hub >= 0.34); falls
    back to the older ``huggingface-cli`` name. Raises if neither is on
    PATH so the user gets a clear hint.
    """
    for name in ("hf", "huggingface-cli"):
        if shutil.which(name) is not None:
            return [name]
    raise click.ClickException(
        "Neither 'hf' nor 'huggingface-cli' is on PATH. "
        "Install it with: pip install -U 'huggingface_hub[cli]'"
    )


def _download(repo_id: str, local_dir):
    cli = _hf_cli()
    cmd = [
        *cli,
        "download",
        repo_id,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(local_dir),
    ]
    _logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


@click.command()
@click.option(
    "--dataset",
    "dataset_name",
    type=click.Choice(sorted(_DATASETS), case_sensitive=False),
    default="mnist",
    show_default=True,
    help="Which dataset to prefetch.",
)
def main(dataset_name: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    repo_id, subdir = _DATASETS[dataset_name]
    target = utils.data_path() / "external" / subdir
    target.mkdir(parents=True, exist_ok=True)

    _logger.info("Fetching %s -> %s", repo_id, target)
    _download(repo_id, target)
    _logger.info("Done. %s is ready.", dataset_name)


if __name__ == "__main__":
    sys.exit(main())
