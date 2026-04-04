"""Create or update a Hugging Face Docker Space for the SafeChat ML service."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parent

MANAGED_FILES = ("Dockerfile", "README.md", "requirements.txt", ".dockerignore")


def clear_bundle_dir(output_dir: Path) -> None:
    """Remove stale bundle contents from previous deployment attempts."""
    if not output_dir.exists():
        return

    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def build_space_bundle(output_dir: Path) -> None:
    """Assemble a minimal Docker Space payload from the local ml-service folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_bundle_dir(output_dir)

    for filename in MANAGED_FILES:
        shutil.copy2(ROOT / filename, output_dir / filename)

    shutil.copytree(
        ROOT / "app",
        output_dir / "app",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Target Hugging Face Space repo id, for example username/safechat-ml-service")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token. Uses local login if omitted.")
    parser.add_argument("--private", action="store_true", help="Create the Space as private.")
    parser.add_argument("--space-hardware", default=None, help="Optional hardware tier, for example cpu-basic.")
    parser.add_argument("--space-storage", default=None, help="Optional persistent storage tier.")
    parser.add_argument("--space-sleep-time", type=int, default=None, help="Optional sleep time in seconds.")
    parser.add_argument(
        "--upload-method",
        choices=("large", "folder"),
        default="folder",
        help="Upload strategy. 'folder' is enough for the smaller Space bundle. 'large' remains available if needed.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Worker count for large-folder upload. Keep this low on slower or less stable networks.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=30,
        help="Seconds between large-folder progress reports.",
    )
    parser.add_argument(
        "--keep-bundle",
        action="store_true",
        help="Keep the generated upload bundle under ml-service/.hf_space_bundle after upload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi(token=args.token)

    bundle_path = ROOT / ".hf_space_bundle"
    workdir = bundle_path

    build_space_bundle(workdir)

    repo_url = api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
        space_hardware=args.space_hardware,
        space_storage=args.space_storage,
        space_sleep_time=args.space_sleep_time,
    )

    if args.upload_method == "large":
        api.upload_large_folder(
            repo_id=args.repo_id,
            repo_type="space",
            folder_path=str(workdir),
            num_workers=args.num_workers,
            print_report=True,
            print_report_every=args.report_every,
        )
    else:
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="space",
            folder_path=str(workdir),
            commit_message="Deploy SafeChat ML service to Hugging Face Space",
        )

    print(f"Uploaded Space bundle to {repo_url}")
    print(f"Space URL: https://huggingface.co/spaces/{args.repo_id}")
    print(f"Bundle path: {workdir}")
    if args.upload_method == "large":
        print("Large-folder upload metadata is stored under the bundle .cache directory for resume support.")


if __name__ == "__main__":
    main()
