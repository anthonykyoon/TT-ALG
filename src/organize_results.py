#!/usr/bin/env python3
"""
organize_results.py -- group result files into one folder per results set.

Each results CSV  <stem>.csv  defines a group, and the folder is labelled with
that file's name (its stem).  Its plots follow the convention written by
plot_results.py,  <metric>_vs_N_<stem>.png  -- i.e. every file whose name ends in
"_<stem>" (any extension) belongs to that CSV.  For each group this script makes
a folder  <stem>/  and COPIES the CSV together with all of its plots into it, so
the originals are left untouched.

A file that matches several stems is assigned to the LONGEST (most specific) one,
so a generic  glauber_results.csv  will not swallow  glauber_results_1_kt_0.3 's
plots.  Files that match no group (e.g. glauber_M_vs_T.png) are reported and left
where they are.

By default it scans the repo root (where a stray sweep CSV lands if it was run
with an explicit path) and files each group into results/<stem>/.

Usage:
    python3 src/organize_results.py              # repo root -> results/<stem>/
    python3 src/organize_results.py --src DIR    # scan a different folder
    python3 src/organize_results.py --dest DIR   # put the group folders elsewhere
    python3 src/organize_results.py --pattern '*.csv'  # what defines a group
    python3 src/organize_results.py --move       # move instead of copy
    python3 src/organize_results.py --force      # overwrite files already copied
    python3 src/organize_results.py --dry-run    # print the plan, touch nothing
"""

import os
import sys
import glob
import shutil
import argparse
import filecmp


def find_groups(src, pattern):
    """Return {stem: definer_filename} for every file in src matching `pattern`."""
    groups = {}
    for path in sorted(glob.glob(os.path.join(src, pattern))):
        if not os.path.isfile(path):
            continue
        fname = os.path.basename(path)
        stem = os.path.splitext(fname)[0]
        groups[stem] = fname
    return groups


def assign(fname, stems, definers):
    """Which group does `fname` belong to?  Longest matching stem wins; a group
    definer only ever belongs to its own group.  Returns a stem or None."""
    stem_of = os.path.splitext(fname)[0]
    # a definer file (e.g. a CSV) stays with the group it defines
    if fname in definers:
        return stem_of if stem_of in stems else None
    # otherwise: the file attaches to stem S if its own stem is S or ends in "_S"
    matches = [s for s in stems if stem_of == s or stem_of.endswith("_" + s)]
    return max(matches, key=len) if matches else None


def plan(src, groups):
    """Build {stem: [filenames]} and a list of unmatched files (top level only)."""
    stems = list(groups)
    definers = set(groups.values())
    buckets = {s: [] for s in stems}
    unmatched = []
    for fname in sorted(os.listdir(src)):
        full = os.path.join(src, fname)
        if not os.path.isfile(full) or fname == os.path.basename(__file__):
            continue
        s = assign(fname, stems, definers)
        if s is None:
            unmatched.append(fname)
        else:
            buckets[s].append(fname)
    return buckets, unmatched


def transfer(src_file, dst_file, move, force):
    """Copy or move one file; skip an identical existing target.  Return a verb."""
    if os.path.exists(dst_file):
        if not force and filecmp.cmp(src_file, dst_file, shallow=False):
            return "skip (identical)"
        if not force:
            return "skip (exists, differs -- use --force)"
    if move:
        shutil.move(src_file, dst_file)
        return "moved"
    shutil.copy2(src_file, dst_file)
    return "copied"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here) if os.path.basename(here) == "src" else here
    ap.add_argument("--src", default=repo, help="folder to scan (default: the repo root)")
    ap.add_argument("--dest", default=os.path.join(repo, "results"),
                    help="where to create the group folders (default: <repo>/results)")
    ap.add_argument("--pattern", default="*.csv", help="glob for group-defining files (default: *.csv)")
    ap.add_argument("--move", action="store_true", help="move files instead of copying them")
    ap.add_argument("--force", action="store_true", help="overwrite a target file even if it already exists")
    ap.add_argument("--dry-run", action="store_true", help="print the plan without creating or moving anything")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dest = os.path.abspath(args.dest)
    if not os.path.isdir(src):
        print(f"source folder not found: {src}")
        sys.exit(1)

    groups = find_groups(src, args.pattern)
    if not groups:
        print(f"no files matching {args.pattern!r} in {src}; nothing to organize.")
        sys.exit(1)

    buckets, unmatched = plan(src, groups)
    verb = "MOVE" if args.move else "COPY"
    print(f"\norganizing {len(groups)} group(s) from  {src}")
    print(f"into folders under  {dest}   ({verb}{', DRY RUN' if args.dry_run else ''})\n")

    total = 0
    for stem, files in buckets.items():
        folder = os.path.join(dest, stem)
        print(f"[{stem}]  ->  {folder}   ({len(files)} file(s))")
        if not args.dry_run:
            os.makedirs(folder, exist_ok=True)
        for fname in files:
            src_file = os.path.join(src, fname)
            dst_file = os.path.join(folder, fname)
            if os.path.abspath(src_file) == os.path.abspath(dst_file):
                print(f"    - {fname:50s} skip (already in place)")
                continue
            if args.dry_run:
                print(f"    - {fname:50s} would {verb.lower()}")
            else:
                print(f"    - {fname:50s} {transfer(src_file, dst_file, args.move, args.force)}")
            total += 1
        print()

    if unmatched:
        print(f"unmatched (left in place): {len(unmatched)} file(s)")
        for fname in unmatched:
            print(f"    - {fname}")
        print()

    print(f"done: {total} file operation(s) across {len(groups)} group(s)"
          f"{' (dry run -- nothing changed)' if args.dry_run else ''}.")


if __name__ == "__main__":
    main()
