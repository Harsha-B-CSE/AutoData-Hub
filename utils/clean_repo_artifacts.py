"""Preview repository-owned artifacts; pass --force to remove them before building.

Run against a quiescent checkout. Tracked files, dependencies, private storage,
models, fixtures, placeholders and filesystem links are never cleanup targets.
"""

import argparse
from fnmatch import fnmatchcase
import os
from pathlib import Path
import stat
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
PRESERVE_FILES = {
    "app.py", "verify_codebase.py", "readme.md", ".env", ".env.example", ".gitignore",
    "requirements.txt", "requirements-cpu.txt", "requirements-gpu.txt", "pytest.ini",
}
EXCLUDED = {
    ".git", ".hg", ".svn", ".kilo", "node_modules", "site-packages", "dist-packages",
    "venv", ".venv", "env", ".env", ".tox", ".nox",
}
CACHE_DIRS = {"__pycache__", ".pytest_cache", "htmlcov", ".hypothesis", ".coverage"}
FRONTEND_CACHE = {"frontend/.vite", "frontend/.vite-temp", "frontend/dist"}
SCRATCH = {"tmp", "tmp/scratch", "storage/scratch"}
EXTENSIONS = {".pyc", ".pyo", ".pyd", ".tmp", ".bak", ".log", ".coverage"}
ARTIFACT_FILES = {".ds_store", "thumbs.db", "desktop.ini", "coverage.xml", ".coverage", "pytest-results.xml"}
ROOT_SCRIPTS = ("tmp_*.py", "test_scratch_*.py", "populate_*.py", "copy_*.ps1")


def _redirected(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def _tracked_files(root: Path) -> set[Path]:
    if any(os.environ.get(key) for key in ("GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")):
        raise RuntimeError("Refusing cleanup with a Git repository or index override in the environment.")
    if not os.path.lexists(root / ".git"):
        return set()  # A source distribution has no Git index.
    git = ["git", "--no-optional-locks", "-C", str(root)]
    worktree = subprocess.check_output([*git, "rev-parse", "--show-toplevel"])
    if Path(os.fsdecode(worktree).rstrip("\r\n")).resolve(strict=True) != root:
        raise RuntimeError("Git worktree does not match the cleanup root.")
    tracked = subprocess.check_output([*git, "ls-files", "--cached", "-z"])
    return {Path(os.fsdecode(name)) for name in tracked.split(b"\0") if name}


def _plan(root: Path) -> list[tuple[Path, os.stat_result]]:
    tracked = _tracked_files(root)
    plan = []

    def visit(directory: Path, purge_contents: bool = False) -> bool:
        """Return whether this directory will be empty after executing the plan."""
        empty = True
        for path in sorted(directory.iterdir(), key=lambda p: (p.name.casefold(), p.name)):
            rel = path.relative_to(root)
            key = rel.as_posix().casefold()
            name = path.name.casefold()
            if (rel in tracked or name in EXCLUDED or name == ".gitkeep"
                    or (directory == root and name in PRESERVE_FILES)
                    or key in {"models", "tests/fixtures"}
                    or (key.startswith("storage/") and rel.parts[1].casefold() != "scratch")):
                empty = False
                continue

            info = path.lstat()
            if _redirected(info):
                empty = False
                continue
            if stat.S_ISDIR(info.st_mode):
                if (name.startswith((".venv-", ".venv_", "venv-", "venv_")) or os.path.ismount(path)
                        or any(os.path.lexists(path / marker) for marker in ("pyvenv.cfg", "conda-meta", ".git"))):
                    empty = False
                    continue
                purge = purge_contents or name in CACHE_DIRS or key in FRONTEND_CACHE or key in SCRATCH
                child_empty = visit(path, purge)
                if purge and key not in SCRATCH and child_empty:
                    plan.append((path, info))
                    continue
            elif stat.S_ISREG(info.st_mode) and (
                purge_contents or path.suffix.casefold() in EXTENSIONS or name in ARTIFACT_FILES
                or name.startswith(".coverage.") or name.endswith("-pytest-results.xml")
                or (directory == root and any(fnmatchcase(name, pattern) for pattern in ROOT_SCRIPTS))
            ):
                plan.append((path, info))
                continue
            empty = False
        return empty

    visit(root)
    return plan


def clean_repository(dry_run: bool = True) -> None:
    """List or remove matching files, then remove only their eligible empty directories."""
    root = ROOT_DIR.resolve(strict=True)
    markers = (root / name for name in ("app.py", "verify_codebase.py"))
    if root == root.parent or not all(path.is_file() and not _redirected(path.lstat()) for path in markers):
        raise ValueError(f"Refusing to clean an unrecognized repository root: {root}")
    plan = _plan(root)  # A scan or Git index failure must happen before the first deletion.
    print(f"AutoData-Hub cleanup | {'DRY RUN' if dry_run else 'FORCE'} | {root}")
    files = directories = size = failures = 0
    for path, original in plan:
        is_dir = stat.S_ISDIR(original.st_mode)
        label = repr(path.relative_to(root).as_posix())
        if not dry_run:
            try:
                current = path.lstat()
                if (_redirected(current) or path.resolve(strict=True) != path
                        or not os.path.samestat(original, current)
                        or (not is_dir and (original.st_size, original.st_mtime_ns)
                            != (current.st_size, current.st_mtime_ns))):
                    raise OSError("entry changed after the scan; refusing to remove it")
                if is_dir:
                    path.rmdir()  # Never recursively remove newly arrived or protected entries.
                else:
                    path.unlink()
            except OSError as exc:
                failures += 1
                print(f"[FAILED] {label}: {exc}", file=sys.stderr)
                continue
        print(f"[{'DIR' if is_dir else 'FILE'}] {label}")
        if is_dir:
            directories += 1
        else:
            files += 1
            size += original.st_size
    print(f"{'Planned' if dry_run else 'Removed'}: {files} files, {directories} directories, "
          f"{size:,} logical bytes ({size / 2**20:.2f} MiB); {failures} errors.")
    if failures:
        raise RuntimeError("Cleanup incomplete; see failed entries above.")
    if dry_run:
        print("No changes made. Pass --force to apply the cleanup.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Delete artifacts instead of previewing them.")
    try:
        clean_repository(dry_run=not parser.parse_args().force)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Cleanup failed: {error}\n")
