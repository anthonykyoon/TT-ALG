#!/usr/bin/env python3
"""
check_tex_sync.py -- verify that the self-contained "Glauber TT" bundle stays in
sync with the master project.  Two kinds of file are checked:

TEX  (a subset relationship):
    glauber_tt.tex  =  summer_project.tex's PREAMBLE
                       +  everything from the Glauber section to \\end{document},
    with only the document TITLE changed.  So the two must agree on:
      1. the PREAMBLE (up to the first shared section), except the \\title{...} line,
      2. the SHARED BODY (the Glauber section through \\end{document}), byte-for-byte.
    The master's extra sections (Intro/Galerkin, Fokker-Planck) live between the
    preamble and the shared body and are not checked.  The tex check is anchor-
    based (keys off marker text, not line numbers), so it survives line shifts.

    Regions (per file):
        preamble    : start .. \\newpage after \\tableofcontents  (checked, modulo title)
        dropped     : preamble end .. shared-body anchor          (NOT checked)
        shared body : shared-body anchor .. \\end{document}       (checked, exact)

PYTHON  (exact copies):
    the solver CODE files the bundle needs to run are byte-identical copies of
    the repo-root originals; each is compared in full.

CONFIG  (presence only):
    glauber_config.py is user-tunable, so it is only required to EXIST in the
    bundle (so the bundle runs); its contents may legitimately differ from the
    root's.  A content difference is reported as an informational note, not drift
    -- exactly as the tex \\title{...} line is an allowed difference.

Exit code 0 = in sync, 1 = drift detected, 2 = structural problem (anchor/title
not found).  Run from anywhere; paths are resolved relative to this file.
"""

import re
import sys
import difflib
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "Glauber TT"
MASTER = HERE / "glauber" / "summer_project.tex"
COPY = BUNDLE / "glauber_tt.tex"

# Solver CODE files: must be byte-identical between the repo root and the bundle.
PY_FILES = [
    "tt_dmrg.py",
    "tt_gmres.py",
    "glauber_mpo.py",
    "glauber_gmres.py",
    "glauber_solve.py",
]

# User-tunable config: must EXIST in the bundle, but its contents may differ
# (a content difference is an informational note, not a sync failure).
CONFIG_FILES = ["glauber_config.py"]

# The line that marks the start of the shared body in BOTH tex files.
BODY_ANCHOR = r"\section{Formulation of Glauber Dynamics"

# The preamble ends at the first \newpage at or after \tableofcontents.
TOC_MARKER = r"\tableofcontents"
PREAMBLE_END = r"\newpage"

# The document title is allowed to differ freely between the two files; any
# \title{...} line is canonicalised before comparison so it never counts as drift.
TITLE_RE = re.compile(r"^\s*\\title\{.*\}\s*$")


def read_lines(path):
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        sys.exit(2)
    return path.read_text().splitlines()


def find_line(lines, marker, path, start=0):
    """Index of the first line at/after `start` containing `marker`, or exit(2)."""
    for i in range(start, len(lines)):
        if marker in lines[i]:
            return i
    print(f"ERROR: marker {marker!r} not found in {path}")
    sys.exit(2)


def split_regions(lines, path):
    """Split a file into (preamble, dropped, body).

    preamble : start .. the \\newpage after \\tableofcontents (inclusive)
    dropped  : preamble end .. body anchor      (master-only extra sections)
    body     : body anchor .. end of file
    """
    toc = find_line(lines, TOC_MARKER, path)
    pre_end = find_line(lines, PREAMBLE_END, path, start=toc)   # \newpage after TOC
    body_start = find_line(lines, BODY_ANCHOR, path, start=pre_end)
    return lines[:pre_end + 1], lines[pre_end + 1:body_start], lines[body_start:]


def normalise_preamble(lines, path):
    """Preamble with the \\title{...} line canonicalised, so the intentional
    title difference does not count as drift.  Any *other* preamble change will."""
    out = []
    saw_title = False
    for ln in lines:
        if TITLE_RE.match(ln):
            out.append("<TITLE>")
            saw_title = True
        else:
            out.append(ln)
    if not saw_title:
        print(f"WARNING: no \\title{{...}} line found in {path.name} "
              "(title exemption did not apply).")
    return out


def report(label, a_lines, b_lines, a_name, b_name):
    """Print a unified diff for one region; return True if identical."""
    if a_lines == b_lines:
        print(f"  [OK]   {label} identical ({len(a_lines)} lines)")
        return True
    diff = list(difflib.unified_diff(
        a_lines, b_lines, fromfile=f"{a_name} ({label})",
        tofile=f"{b_name} ({label})", lineterm=""))
    print(f"  [DRIFT] {label} differs:")
    for d in diff:
        print("    " + d)
    return False


def check_tex():
    """Check the tex subset relationship; return True if in sync."""
    master_lines = read_lines(MASTER)
    copy_lines = read_lines(COPY)

    m_pre, m_drop, m_body = split_regions(master_lines, MASTER)
    c_pre, c_drop, c_body = split_regions(copy_lines, COPY)

    print("TEX  (subset: preamble + shared body)")
    print(f"  master : {MASTER}")
    print(f"  copy   : {COPY}")
    print(f"  (master drops {len(m_drop)} lines of extra sections before the "
          f"shared body; copy drops {len(c_drop)})")

    ok = True
    ok &= report("preamble (modulo title)",
                 normalise_preamble(m_pre, MASTER),
                 normalise_preamble(c_pre, COPY),
                 "summer_project.tex", "glauber_tt.tex")
    ok &= report("shared body", m_body, c_body,
                 "summer_project.tex", "glauber_tt.tex")
    return ok


def check_python():
    """Byte-identical check of the code files; return True if in sync."""
    print("\nPYTHON  (exact copies: repo root -> bundle)")
    ok = True
    for name in PY_FILES:
        root = HERE / name
        bundled = BUNDLE / name
        missing = [p for p in (root, bundled) if not p.exists()]
        if missing:
            print(f"  [MISSING] {name}: not found at " +
                  ", ".join(str(p) for p in missing))
            ok = False
            continue
        a = root.read_text().splitlines()
        b = bundled.read_text().splitlines()
        ok &= report(name, a, b, f"root/{name}", f"bundle/{name}")
    return ok


def check_config():
    """Presence check of user-tunable config; content differences are informational."""
    print("\nCONFIG  (presence required; contents may differ, user-tunable)")
    ok = True
    for name in CONFIG_FILES:
        root = HERE / name
        bundled = BUNDLE / name
        if not bundled.exists():
            print(f"  [MISSING] {name}: absent from the bundle "
                  "(bundle would not run)")
            ok = False
            continue
        if root.exists() and root.read_text() == bundled.read_text():
            print(f"  [OK]   {name} present and identical")
        else:
            print(f"  [note] {name} present but differs from root "
                  "(allowed: user-tunable config)")
    return ok


def main():
    ok = check_tex()
    ok &= check_python()
    ok &= check_config()

    print()
    if ok:
        print("IN SYNC: tex shared content (title aside), all bundle code files "
              "match, and the config is present.")
        sys.exit(0)
    print("OUT OF SYNC: reconcile the differences above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
