import importlib.util
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "utils" / "clean_repo_artifacts.py"


def seed(root, *names):
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns) if p.is_file() else None
        for p in root.rglob("*")
    }


@pytest.fixture
def script(tmp_path, monkeypatch):
    # Git invoked from a hook can inherit the caller's worktree or index.
    for key in tuple(os.environ):
        if key.startswith("GIT_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    # Even ignoring the ROOT_DIR override must not point a force run at the real checkout.
    path = tmp_path / "utils" / SOURCE.name
    path.parent.mkdir()
    return shutil.copyfile(SOURCE, path)


@pytest.fixture
def checkout(tmp_path, script, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("sandbox_cleaner", script)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    root = tmp_path / "repo"
    seed(root, "app.py", "verify_codebase.py")
    monkeypatch.setattr(module, "ROOT_DIR", root)
    return root, module.clean_repository


def test_default_preview_order_and_force_idempotence(checkout, capsys, monkeypatch):
    root, clean = checkout
    targets = ["zulu.pyc", "middle/bytecode.pyo", "alpha.pyc"]
    seed(root, *targets, "middle/source.py")
    before = snapshot(root)
    # A naturally sorted filesystem (notably NTFS) would otherwise mask a missing sort.
    iterdir = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda p: reversed(sorted(iterdir(p))))

    def reported(output):
        return [name for line in output.replace("\\", "/").splitlines() for name in targets if name in line]

    clean()
    preview = capsys.readouterr().out
    clean()
    assert capsys.readouterr().out == preview
    assert snapshot(root) == before
    assert reported(preview) == sorted(targets)

    expected = {k: v for k, v in before.items() if k not in targets}
    clean(dry_run=False)
    assert snapshot(root) == expected
    assert reported(capsys.readouterr().out) == sorted(targets)
    clean(dry_run=False)
    assert snapshot(root) == expected
    assert reported(capsys.readouterr().out) == []


def test_artifact_patterns_and_directory_boundaries(checkout):
    root, clean = checkout
    scripts = "tmp_probe.py test_scratch_trial.py populate_sample.py copy_stubs.ps1".split()
    delete = scripts + """
        pkg/__pycache__/entry.bin pkg/.pytest_cache/v/cache/nodeids .hypothesis/examples/db
        htmlcov/index.html reports/coverage.xml .coverage .coverage.worker pkg/foo-pytest-results.xml
        pkg/extension.pyd pkg/Thumbs.db assets/.DS_Store docs/Desktop.ini scratch.bak trace.log work.tmp
        frontend/.vite/deps/bundle.js frontend/dist/assets/app.js tmp/job/result.csv storage/scratch/job/chunk.dat
        cache/.pytest_cache/pinned/deep/stale.json
    """.split()
    keep = [f"scripts/{name}" for name in scripts] + """
        copy_stubs.py setup_project.py test_scratch.py populate.py keep_tmp_probe.py copy_stubs.psm1 deploy.ps1
        tests/test_source.py dist/index.js .vite/source.js frontend/nested/dist/index.js nested/frontend/.vite/user.js
        nested/frontend/dist/user.js frontend/dist_backup/source.js
        cache/.pytest_cache/pinned/deep/.gitkeep .env .env.example .streamlit/config.toml frontend/vite.config.js
    """.split()
    seed(root, *delete, *keep)
    clean(dry_run=False)

    for name in delete + "pkg/__pycache__ pkg/.pytest_cache .hypothesis htmlcov frontend/.vite frontend/dist".split():
        assert not (root / name).exists(), name
    for name in keep:
        assert (root / name).read_bytes() == name.encode()
    for name in ("tmp", "storage/scratch"):
        assert list((root / name).iterdir()) == []


def test_protected_trees_and_git_index_take_priority(checkout, monkeypatch):
    root, clean = checkout
    subprocess.run(["git", "init", "-q", "--template="], cwd=root, check=True)
    tracked = ("tmp_committed.py", "frontend/dist/pinned/old bundle.js")
    keep = """
        .git/probe.pyc models/image_restoration/weights.onnx models/image_restoration/cache.pyc
        storage/models/weights.tmp tests/fixtures/__pycache__/example.pyc frontend/node_modules/pkg/debug.log
        venv/Lib/cache.pyc .venv-gpu/Lib/cache.pyc tmp/custom_runtime/pyvenv.cfg tmp/custom_runtime/lib/cache.pyc
        storage/raw/upload.csv storage/raw/upload.bak storage/intermediate/job/work.tmp storage/processed/export.log
        storage/autodata_hub.db storage/scratch/pinned/nested/.gitkeep
    """.split()
    seed(root, *tracked, *keep, "frontend/dist/pinned/new.js", "tmp_disposable.py")
    subprocess.run(["git", "add", "-f", "--", *tracked], cwd=root, check=True)
    alternate = shutil.copyfile(root / ".git/index", root / ".git/test-index")
    with monkeypatch.context() as env:
        env.setenv("GIT_INDEX_FILE", str(alternate))
        before = snapshot(root)
        with pytest.raises(RuntimeError):
            clean(dry_run=False)
        assert snapshot(root) == before
    clean(dry_run=False)

    for name in (*tracked, *keep):
        assert (root / name).read_bytes() == name.encode()
    assert not (root / "frontend/dist/pinned/new.js").exists()
    assert not (root / "tmp_disposable.py").exists()


def test_directory_link_is_not_a_cleanup_target(checkout, monkeypatch):
    root, clean = checkout
    target = root / "tests/fixtures/sample"
    seed(target, "probe.pyc")
    seed(root, "pkg/ordinary.pyc")
    link = root / "pkg/__pycache__"
    try:
        if os.name == "nt":
            from _winapi import CreateJunction

            CreateJunction(str(target), str(link))
        else:
            link.symlink_to(target, target_is_directory=True)
    except (ImportError, NotImplementedError, OSError):
        # Filesystems without directory links still exercise a reparse-point entry.
        seed(link, "probe.pyc")
        lstat = Path.lstat
        redirect = SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
        monkeypatch.setattr(Path, "lstat", lambda p: redirect if p == link else lstat(p))

    clean(dry_run=False)
    assert not (root / "pkg/ordinary.pyc").exists()
    assert (link / "probe.pyc").read_bytes() == (target / "probe.pyc").read_bytes() == b"probe.pyc"


def test_failed_removal_is_not_reported_as_success(checkout, monkeypatch, capsys):
    root, clean = checkout
    seed(root, "busy.pyc", "other.pyc")
    unlink = Path.unlink

    def reject_busy(path, *args, **kwargs):
        if path == root / "busy.pyc":
            raise PermissionError("fixture is busy")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", reject_busy)
    with pytest.raises(RuntimeError):
        clean(dry_run=False)
    captured = capsys.readouterr()
    assert (root / "busy.pyc").read_bytes() == b"busy.pyc"
    assert not (root / "other.pyc").exists()
    assert "busy.pyc" in captured.err and "busy.pyc" not in captured.out
    assert "other.pyc" in captured.out
    assert re.search(r"\b1 files?\b", captured.out) and re.search(r"\b1 errors?\b", captured.out)


def test_cli_defaults_to_preview_and_requires_repo_markers(tmp_path, script):
    seed(tmp_path, "app.py", "verify_codebase.py", "stale.pyc")
    cmd = [sys.executable, "-B", str(script)]
    cwd = tmp_path / "utils"
    subprocess.run(cmd, cwd=cwd, capture_output=True, check=True)
    assert (tmp_path / "stale.pyc").read_bytes() == b"stale.pyc"
    subprocess.run([*cmd, "--force"], cwd=cwd, capture_output=True, check=True)
    assert not (tmp_path / "stale.pyc").exists()

    (tmp_path / "verify_codebase.py").unlink()
    seed(tmp_path, "stale.pyc")
    assert subprocess.run([*cmd, "--force"], cwd=cwd, capture_output=True).returncode != 0
    assert (tmp_path / "stale.pyc").read_bytes() == b"stale.pyc"
