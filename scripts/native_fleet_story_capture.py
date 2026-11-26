"""Keep the existing Explorer read-only capture alive for one attended story."""

import argparse
import json
from pathlib import Path
import time

from explorer4 import fleet_capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--seconds", type=int, choices=(15, 30, 60), default=60)
    args = parser.parse_args()
    result = fleet_capture.start(args.seconds)
    temp = args.receipt.with_suffix(".pending")
    temp.write_text(json.dumps(result))
    temp.replace(args.receipt)
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline and fleet_capture.state()["active"] == result["id"]:
        time.sleep(.5)
    outcome = json.loads((fleet_capture.entry(result["id"]) / "manifest.json").read_text())
    if outcome["phase"] != "complete":
        raise RuntimeError(outcome.get("error") or "Explorer native capture has no complete manifest")


if __name__ == "__main__":
    main()
