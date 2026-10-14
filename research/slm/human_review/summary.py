"""Read-only summary of actual saved human decisions."""
import json
from review_core import summarize

if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
