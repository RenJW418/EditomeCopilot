"""
Upload large data files to Hugging Face Hub dataset repository.

Usage:
    python scripts/upload_data_to_hf.py --token hf_xxxx --repo YOUR_HF_USERNAME/editome-copilot-data

The following files will be uploaded:
    data/faiss_db/index.faiss           (~254 MB)
    data/faiss_db/index.pkl             (~166 MB)
    data/faiss_db/bm25_corpus.pkl       (~163 MB)
    data/knowledge_base/literature_db_GEA_v2026_Q1.json  (~168 MB)
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

FILES_TO_UPLOAD = [
    "data/faiss_db/index.faiss",
    "data/faiss_db/index.pkl",
    "data/faiss_db/bm25_corpus.pkl",
    "data/knowledge_base/literature_db_GEA_v2026_Q1.json",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="Hugging Face write token (hf_...)")
    parser.add_argument(
        "--repo",
        required=True,
        help="HF dataset repo id, e.g. RenJW418/editome-copilot-data",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    api = HfApi(token=args.token)

    # Create repo if it doesn't exist
    print(f"[1/2] Creating dataset repo: {args.repo}")
    create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=False,
        exist_ok=True,
        token=args.token,
    )

    # Upload files
    print("[2/2] Uploading files ...")
    for rel_path in FILES_TO_UPLOAD:
        local = root / rel_path
        if not local.exists():
            print(f"  SKIP (not found): {rel_path}")
            continue
        size_mb = local.stat().st_size / 1_048_576
        print(f"  Uploading {rel_path}  ({size_mb:.1f} MB) ...")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=rel_path,
            repo_id=args.repo,
            repo_type="dataset",
            commit_message=f"Upload {rel_path}",
        )
        print(f"  ✓ Done: https://huggingface.co/datasets/{args.repo}/blob/main/{rel_path}")

    print("\nAll done!")
    print(f"Dataset URL: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
