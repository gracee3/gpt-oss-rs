#!/usr/bin/env python3
"""Reject broken repository-relative inline Markdown links."""

from __future__ import annotations

import re
import subprocess
import sys
import urllib.parse
from pathlib import Path


LINK = re.compile(r"!?\[[^\]]*\]\((?:<([^>]+)>|([^\s)]+))(?:\s+[^)]*)?\)")


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    tracked = sorted(
        set(
            subprocess.check_output(
                [
                    "git",
                    "ls-files",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "*.md",
                ],
                cwd=repository,
                text=True,
            ).splitlines()
        )
    )
    failures = []
    for relative in tracked:
        source = repository / relative
        for line_number, line in enumerate(source.read_text(errors="replace").splitlines(), 1):
            for match in LINK.finditer(line):
                target = (match.group(1) or match.group(2)).split("#", 1)[0]
                target = urllib.parse.unquote(target)
                if not target or target.startswith(("http://", "https://", "mailto:", "data:")):
                    continue
                path = repository / target.removeprefix("/") if target.startswith("/") else source.parent / target
                if not path.exists():
                    failures.append(f"{relative}:{line_number}: missing {target}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"validated relative Markdown links in {len(tracked)} repository files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
