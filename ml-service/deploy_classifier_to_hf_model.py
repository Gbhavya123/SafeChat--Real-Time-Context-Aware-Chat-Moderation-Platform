"""Upload the fine-tuned MuRIL checkpoint to a Hugging Face model repo."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "checkpoints" / "muril-toxicity-finetuned"
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_args.bin",
)


def build_model_bundle(output_dir: Path, repo_id: str) -> None:
    """Create a minimal model repo bundle that can be resumed safely."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for child in output_dir.iterdir():
        if child.name == ".cache":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    for filename in MODEL_FILES:
        shutil.copy2(SOURCE_DIR / filename, output_dir / filename)

    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "---",
                "license: mit",
                "library_name: transformers",
                "pipeline_tag: text-classification",
                "---",
                "",
                "# SafeChat MuRIL Toxicity Classifier",
                "",
                f"Model repo: `{repo_id}`",
                "",
                "Fine-tuned MuRIL checkpoint used by the SafeChat ML service for multilingual toxicity classification.",
                "",
                "Labels:",
                "- toxic",
                "- severe_toxic",
                "- obscene",
                "- threat",
                "- insult",
                "- identity_hate",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Target Hugging Face model repo id, for example username/safechat-muril-toxicity-finetuned")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token. Uses local login if omitted.")
    parser.add_argument("--private", action="store_true", help="Create the repo as private.")
    parser.add_argument("--num-workers", type=int, default=1, help="Worker count for large-folder upload.")
    parser.add_argument("--report-every", type=int, default=30, help="Seconds between progress reports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi(token=args.token)

    bundle_path = ROOT / ".hf_model_bundle"
    build_model_bundle(bundle_path, args.repo_id)

    repo_url = api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(bundle_path),
        num_workers=args.num_workers,
        print_report=True,
        print_report_every=args.report_every,
    )

    print(f"Uploaded model bundle to {repo_url}")
    print(f"Model URL: https://huggingface.co/{args.repo_id}")
    print(f"Bundle path: {bundle_path}")
    print("Large-folder upload metadata is stored under the bundle .cache directory for resume support.")


if __name__ == "__main__":
    main()
