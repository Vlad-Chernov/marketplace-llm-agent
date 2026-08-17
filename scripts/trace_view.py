import argparse
import json
from pathlib import Path


def main() -> None:
    """Print a compact view of JSONL trace events."""

    parser = argparse.ArgumentParser()
    parser.add_argument("trace_path", type=Path)
    args = parser.parse_args()

    for line in args.trace_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        print(
            f"{event['timestamp']} | {event['run_id']} | "
            f"{event['event_type']} | {event['payload']}"
        )


if __name__ == "__main__":
    main()