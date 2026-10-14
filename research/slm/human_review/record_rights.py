"""Record a human rights-review event; never infer approval from metadata."""
import argparse
import json
from rights_core import append_rights


def main():
    parser = argparse.ArgumentParser(description="Append a reviewed per-paper rights decision")
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--decision", required=True, choices=("permissive_verified", "restricted", "unresolved"))
    parser.add_argument("--source-version-matches", required=True, choices=("Yes", "No", "Uncertain"))
    parser.add_argument("--third-party-material-cleared", required=True, choices=("Yes", "No", "Uncertain"))
    parser.add_argument("--training-and-redistribution-allowed", required=True, choices=("Yes", "No", "Uncertain"))
    parser.add_argument("--license-evidence-url", required=True)
    parser.add_argument("--attribution", required=True)
    parser.add_argument("--notes", required=True)
    args = parser.parse_args()
    decision = {key: value for key, value in vars(args).items()}
    print(json.dumps(append_rights(decision), indent=2))


if __name__ == "__main__":
    main()
