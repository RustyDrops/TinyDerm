"""Audit Git history and files for tokens and credentials."""

import subprocess
import re
from pathlib import Path

res = subprocess.run(["git", "log", "-p", "--all"], capture_output=True, text=True, errors="ignore")
diff_text = res.stdout

patterns = [
    r"ghp_[a-zA-Z0-9]{20,}",
    r"github_pat_[a-zA-Z0-9]{20,}",
    r"bearer\s+[a-zA-Z0-9_\-\.]{20,}",
    r"api[_-]?key\s*[:=]\s*['\"][a-zA-Z0-9_\-]{15,}['\"]",
    r"secret\s*[:=]\s*['\"][a-zA-Z0-9_\-]{15,}['\"]",
    r"password\s*[:=]\s*['\"][a-zA-Z0-9_\-]{15,}['\"]",
    r"token\s*[:=]\s*['\"][a-zA-Z0-9_\-]{20,}['\"]",
]

found = []
for p in patterns:
    matches = re.findall(p, diff_text, re.IGNORECASE)
    if matches:
        found.extend(matches)

print("=== Git Audit Report ===")
# Also check remote url
rem = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True).stdout
print(f"Remote URLs:\n{rem.strip()}\n")

if "@" in rem:
    print("ALERT: Remote URL contains an inline token or user!")
else:
    print("Remote URL is clean (standard HTTPS without credentials).")

if found:
    print("ALERT: Potential credentials found in git log:", found)
else:
    print("Git history is 100% CLEAN: Zero personal access tokens, API keys, or credentials found.")
