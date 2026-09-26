"""Run persistent local timer/alarm delivery without Ollama or Moonshine."""

import argparse
from time import sleep

from local_schedule import Schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.1 <= args.interval <= 60:
        parser.error("--interval must be between 0.1 and 60 seconds")
    schedule = Schedule()
    print("Local timer/alarm scheduler running. Press Ctrl+C to stop.", flush=True)
    try:
        while True:
            for message in schedule.fire_due():
                print(f"\n[Schedule] {message}", flush=True)
            sleep(args.interval)
    except KeyboardInterrupt:
        print("\nScheduler stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
