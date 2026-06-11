# SPDX-License-Identifier: Apache-2.0
"""Minimal issue-tracker clients: plain REST + a token, no SDKs.

GitHub:  GITHUB_TOKEN          + LOGWATCH_GITHUB_REPO  (owner/repo)
GitLab:  GITLAB_TOKEN          + LOGWATCH_GITLAB_PROJECT (id or url-encoded path)
                               + LOGWATCH_GITLAB_URL (default https://gitlab.com)

Every issue carries the `logwatch` label and a fenced metadata block the
feedback loop parses back out. Issue outcome conventions:
  closed as completed / state_reason "completed"      -> finding was real
  closed as not planned / label `logwatch-rejected`   -> false positive
  comment `correction: <label>` on a rejected issue   -> supervised correction
"""

import json
import os
import re

import requests

METADATA_FENCE = "```logwatch-metadata"
_METADATA_RE = re.compile(r"```logwatch-metadata\s*\n(.*?)```", re.S)
_CORRECTION_RE = re.compile(r"^correction:\s*(\w+)", re.I | re.M)


def issue_body(finding: dict) -> str:
    metadata = {
        key: finding[key]
        for key in (
            "fingerprint", "label", "evidence", "chunk_sha",
            "model_version", "count", "log_path", "first_line",
        )
    }
    locations = "\n".join(f"- `{loc}`" for loc in finding["locations"])
    return (
        f"**{finding['label']}** seen {finding['count']}x — {finding['reason']}\n\n"
        f"```\n{finding['evidence']}\n```\n\n"
        f"Locations:\n{locations}\n\n"
        "_Filed by logwatch. Close as **completed** if real (grades the model), "
        "close as **not planned** if a false positive (corrects the model). "
        "Optionally comment `correction: <label>` with the right label._\n\n"
        f"{METADATA_FENCE}\n{json.dumps(metadata, indent=2)}\n```\n"
    )


def parse_metadata(body: str) -> dict | None:
    match = _METADATA_RE.search(body or "")
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def parse_correction(text: str) -> str | None:
    match = _CORRECTION_RE.search(text or "")
    return match.group(1).lower() if match else None


class GitHubTracker:
    def __init__(self):
        self.repo = os.environ["LOGWATCH_GITHUB_REPO"]
        token = os.environ["GITHUB_TOKEN"]
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {token}",
             "Accept": "application/vnd.github+json"}
        )
        self.api = f"https://api.github.com/repos/{self.repo}"

    def _get_all(self, url: str, **params) -> list[dict]:
        results: list[dict] = []
        params = {"per_page": 100, **params}
        while url:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            results.extend(resp.json())
            url = resp.links.get("next", {}).get("url")
            params = {}
        return results

    def existing_fingerprints(self) -> set[str]:
        issues = self._get_all(f"{self.api}/issues",
                               state="all", labels="logwatch")
        found = set()
        for issue in issues:
            meta = parse_metadata(issue.get("body"))
            if meta:
                found.add(meta["fingerprint"])
        return found

    def create_issue(self, finding: dict) -> str:
        resp = self.session.post(
            f"{self.api}/issues",
            json={
                "title": f"[logwatch] {finding['label']}: "
                         f"{finding['evidence'][:80]}",
                "body": issue_body(finding),
                "labels": ["logwatch", f"logwatch:{finding['label']}"],
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["html_url"]

    def closed_issues(self) -> list[dict]:
        """Normalized closed issues: metadata, outcome, optional correction."""
        issues = self._get_all(f"{self.api}/issues",
                               state="closed", labels="logwatch")
        results = []
        for issue in issues:
            meta = parse_metadata(issue.get("body"))
            if not meta:
                continue
            label_names = {lbl["name"] for lbl in issue.get("labels", [])}
            rejected = (
                issue.get("state_reason") == "not_planned"
                or "logwatch-rejected" in label_names
            )
            correction = None
            if rejected:
                comments = self._get_all(issue["comments_url"])
                for comment in comments:
                    correction = parse_correction(comment.get("body")) or correction
            results.append({
                "ref": issue["html_url"],
                "metadata": meta,
                "outcome": "rejected" if rejected else "addressed",
                "correction": correction,
                "closed_at": issue.get("closed_at"),
            })
        return results


class GitLabTracker:
    def __init__(self):
        base = os.environ.get("LOGWATCH_GITLAB_URL", "https://gitlab.com")
        project = os.environ["LOGWATCH_GITLAB_PROJECT"].replace("/", "%2F")
        self.api = f"{base.rstrip('/')}/api/v4/projects/{project}"
        self.session = requests.Session()
        self.session.headers.update(
            {"PRIVATE-TOKEN": os.environ["GITLAB_TOKEN"]}
        )

    def _get_all(self, url: str, **params) -> list[dict]:
        results: list[dict] = []
        params = {"per_page": 100, "page": 1, **params}
        while True:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            batch = resp.json()
            results.extend(batch)
            next_page = resp.headers.get("X-Next-Page")
            if not next_page:
                return results
            params["page"] = next_page

    def existing_fingerprints(self) -> set[str]:
        issues = self._get_all(f"{self.api}/issues",
                               labels="logwatch", state="all")
        found = set()
        for issue in issues:
            meta = parse_metadata(issue.get("description"))
            if meta:
                found.add(meta["fingerprint"])
        return found

    def create_issue(self, finding: dict) -> str:
        resp = self.session.post(
            f"{self.api}/issues",
            json={
                "title": f"[logwatch] {finding['label']}: "
                         f"{finding['evidence'][:80]}",
                "description": issue_body(finding),
                "labels": f"logwatch,logwatch:{finding['label']}",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["web_url"]

    def closed_issues(self) -> list[dict]:
        issues = self._get_all(f"{self.api}/issues",
                               labels="logwatch", state="closed")
        results = []
        for issue in issues:
            meta = parse_metadata(issue.get("description"))
            if not meta:
                continue
            rejected = "logwatch-rejected" in issue.get("labels", [])
            correction = None
            if rejected:
                notes = self._get_all(
                    f"{self.api}/issues/{issue['iid']}/notes")
                for note in notes:
                    correction = parse_correction(note.get("body")) or correction
            results.append({
                "ref": issue["web_url"],
                "metadata": meta,
                "outcome": "rejected" if rejected else "addressed",
                "correction": correction,
                "closed_at": issue.get("closed_at"),
            })
        return results


def get_tracker(name: str):
    if name == "github":
        return GitHubTracker()
    if name == "gitlab":
        return GitLabTracker()
    raise ValueError(f"unknown tracker: {name} (use github or gitlab)")
