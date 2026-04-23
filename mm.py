#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
from pathlib import Path

MODEL_DIR = Path.home() / ".mm"


def pull(repo_model: str) -> None:
    if not repo_model.startswith("hf:"):
        raise ValueError("Model ID must start with 'hf:', e.g., hf:repo/model")

    model_id = repo_model[3:]  # Remove 'hf:' prefix
    parts = model_id.split("/")
    if len(parts) != 2:
        raise ValueError("Model ID must be in format 'repo/model'")

    repo, model = parts
    target_dir = MODEL_DIR / repo / model

    if target_dir.exists():
        print(f"Model {model_id} already installed at {target_dir}")
        return

    target_dir.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://huggingface.co/{model_id}"
    print(f"Downloading {model_id} to {target_dir}...")

    # Clone without LFS to show proper progress
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    subprocess.run(["git", "clone", "--progress", url, str(target_dir)], check=True, env=env)

    # Pull LFS files with proper progress display
    print("Fetching model files...")
    subprocess.run(["git", "lfs", "pull"], cwd=target_dir, check=True)

    print("Removing .git/lfs directory (prune history)")
    subprocess.run(["rm", "-rf", ".git/lfs"], cwd=target_dir, check=True)

    print(f"Successfully installed {model_id}")


def ls() -> None:
    if not MODEL_DIR.exists():
        print("No models installed.")
        return

    models = []
    for repo_dir in sorted(MODEL_DIR.iterdir()):
        if repo_dir.is_dir():
            for model_dir in sorted(repo_dir.iterdir()):
                if model_dir.is_dir():
                    models.append(f"{repo_dir.name}/{model_dir.name}")

    if not models:
        print("No models installed.")
    else:
        for model in models:
            print(model)


def rm(repo_model: str) -> None:
    parts = repo_model.split("/")
    if len(parts) != 2:
        raise ValueError("Model must be in format 'repo/model'")

    repo, model = parts
    target_dir = MODEL_DIR / repo / model

    if not target_dir.exists():
        raise FileNotFoundError(f"Model {repo_model} not found at {target_dir}")

    response = input(f"Remove {repo_model}? (y/N): ").strip().lower()
    if response != "y":
        print("Cancelled.")
        return

    shutil.rmtree(target_dir)
    print(f"Removed {repo_model}")


def main():
    parser = argparse.ArgumentParser(description="Manage HuggingFace models locally")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    pull_parser = subparsers.add_parser("pull", help="Download a model")
    pull_parser.add_argument("model", help="Model ID (e.g., hf:repo/model)")

    subparsers.add_parser("ls", help="List installed models")

    rm_parser = subparsers.add_parser("rm", help="Remove a model")
    rm_parser.add_argument("model", help="Model to remove (e.g., repo/model)")

    args = parser.parse_args()

    if args.command == "pull":
        pull(args.model)
    elif args.command == "ls":
        ls()
    elif args.command == "rm":
        rm(args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
