# SPDX-License-Identifier: Apache-2.0
"""logwatch CLI.

  python3 -m logwatch triage <paths...> [--engine llm|heuristic] [--out FILE]
  python3 -m logwatch file-issues <paths...> [--tracker github|gitlab] [--dry-run]
  python3 -m logwatch feedback [--tracker github|gitlab]

triage      sweep logs, print findings (and write all verdicts to --out)
file-issues triage + open one tracker issue per new actionable finding
feedback    fold closed-issue outcomes into corrections/grades/metrics
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from .feedback import process_closed_issues
from .triage import reduce_findings, triage_paths
from .trackers import get_tracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("logwatch")


def _expand(paths: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.log")))
        else:
            expanded.append(path)
    return expanded


def cmd_triage(args) -> list[dict]:
    verdicts = list(triage_paths(_expand(args.paths), engine=args.engine))
    if args.out:
        with open(args.out, "w") as fh:
            for v in verdicts:
                fh.write(v.to_json() + "\n")
        logger.info("wrote %d verdicts to %s", len(verdicts), args.out)
    findings = reduce_findings(verdicts)
    for finding in findings:
        print(f"[{finding['label']}] x{finding['count']}  "
              f"{finding['evidence'][:100]}")
    logger.info("%d chunks -> %d actionable findings",
                len(verdicts), len(findings))
    return findings


def cmd_file_issues(args) -> None:
    findings = cmd_triage(args)
    if not findings:
        logger.info("nothing actionable, no issues to file")
        return
    if args.dry_run:
        for finding in findings:
            print(f"DRY-RUN would file: [logwatch] {finding['label']}: "
                  f"{finding['evidence'][:80]}")
        return
    tracker = get_tracker(args.tracker)
    known = tracker.existing_fingerprints()
    filed = 0
    for finding in findings:
        if finding["fingerprint"] in known:
            logger.info("skip (already filed): %s", finding["fingerprint"])
            continue
        url = tracker.create_issue(finding)
        logger.info("filed %s", url)
        filed += 1
    logger.info("filed %d new issues (%d duplicates skipped)",
                filed, len(findings) - filed)


def cmd_feedback(args) -> None:
    tracker = get_tracker(args.tracker)
    summary = process_closed_issues(tracker.closed_issues())
    print(json.dumps(summary, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(prog="logwatch")
    sub = parser.add_subparsers(dest="command", required=True)

    p_triage = sub.add_parser("triage")
    p_triage.add_argument("paths", nargs="+")
    p_triage.add_argument("--engine", choices=["heuristic", "llm"],
                          default="heuristic")
    p_triage.add_argument("--out")
    p_triage.set_defaults(func=cmd_triage)

    p_file = sub.add_parser("file-issues")
    p_file.add_argument("paths", nargs="+")
    p_file.add_argument("--engine", choices=["heuristic", "llm"],
                        default="heuristic")
    p_file.add_argument("--out")
    p_file.add_argument("--tracker", choices=["github", "gitlab"],
                        default="github")
    p_file.add_argument("--dry-run", action="store_true")
    p_file.set_defaults(func=cmd_file_issues)

    p_feedback = sub.add_parser("feedback")
    p_feedback.add_argument("--tracker", choices=["github", "gitlab"],
                            default="github")
    p_feedback.set_defaults(func=cmd_feedback)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
