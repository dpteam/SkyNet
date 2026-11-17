#!/usr/bin/env python3
"""ci_status.py - show the latest GitHub Actions run for this repository.

Uses the public API, so it works without gh CLI or a token for public
repositories.  Rate limited to 60 requests per hour unauthenticated, which
is plenty for checking a run by hand.
"""
import json
import sys
import urllib.request

REPO = "dpteam/SkyNet"


def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "skynet-ci-status"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    runs = get("https://api.github.com/repos/%s/actions/runs?per_page=%d" % (REPO, limit))
    if "workflow_runs" not in runs:
        print("API error:", runs)
        return 1

    for run in runs["workflow_runs"]:
        print("run #%s  %s  %s/%s  %s" % (
            run["run_number"], run["head_sha"][:8],
            run["status"], run["conclusion"], run["head_commit"]["message"].split("\n")[0][:60]))

        jobs = get("https://api.github.com/repos/%s/actions/runs/%s/jobs"
                   % (REPO, run["id"]))
        for job in jobs.get("jobs", []):
            concl = job["conclusion"] or job["status"]
            mark = {"success": "ok  ", "failure": "FAIL", "skipped": "skip",
                    "cancelled": "cncl", "in_progress": "....", "queued": "wait"}.get(concl, concl)
            print("   %s %-42s %s" % (mark, job["name"], concl))
            if job["conclusion"] == "failure":
                for step in job.get("steps", []):
                    if step.get("conclusion") == "failure":
                        print("        failed step: %s" % step["name"])
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
