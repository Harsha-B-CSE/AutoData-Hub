import io
import sys
import tempfile
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pytest
import pandas as pd
from core.dataset_manager import DatasetManager

def test_scenario_c_tier1_immutability():
    manager = DatasetManager()
    df_raw = pd.DataFrame({"col1":[1,2,3], "col2": ["A", "B", "C"]})
    manager.register_version("v0_raw", df_raw)
    df_retrieved = manager.get_version("v0_raw")
    assert id(df_raw) != id(df_retrieved)
    df_retrieved.loc[0, "col1"] = 99
    assert manager.get_version("v0_raw")["col1"].iloc[0] == 1


@pytest.mark.parametrize("name", ["preview.bin", "batch/preview.bin", r"batch\preview.bin"])
def test_scratch_uses_only_upload_basename(tmp_path, name):
    manager = DatasetManager(tmp_path / "scratch", tmp_path / "output")
    source = io.BytesIO(b"prefix:data")
    source.seek(7)
    target = manager.save_scratch_file(name, source)
    assert target == manager.get_scratch_dir().resolve() / "preview.bin"
    assert target == manager.get_scratch_file(name)
    assert target.read_bytes() == b"data" and source.tell() == 7
    manager.save_scratch_file(name, "text")
    assert target.read_bytes() == b"text"


@pytest.mark.parametrize("name", ["", ".", "..", "NUL", "report.txt.", "report.txt:notes"])
def test_scratch_requires_a_regular_file_name(tmp_path, name):
    manager = DatasetManager(tmp_path / "scratch", tmp_path / "output")
    with pytest.raises(ValueError, match="^Path traversal security violation$"):
        manager.save_scratch_file(name, b"preview")
    assert not list(manager.get_scratch_dir().iterdir())


def test_scratch_compares_canonical_components_not_text_prefix(tmp_path, monkeypatch):
    root = tmp_path.resolve() / "scratch"
    manager = DatasetManager(root, tmp_path / "output")
    resolve = Path.resolve

    def resolved(path, *args, **kwargs):
        if path == root / "preview.bin":
            return tmp_path / "scratch-archive" / "preview.bin"
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolved)
    with pytest.raises(ValueError, match="^Path traversal security violation$"):
        manager.get_scratch_file("preview.bin")


@pytest.mark.parametrize(("prefix", "would_block"), [(b"partial", False), (b"partial", True), (b"", True)])
def test_incomplete_stream_cannot_replace_scratch_file(tmp_path, prefix, would_block):
    manager = DatasetManager(tmp_path / "scratch", tmp_path / "output")
    target = manager.save_scratch_file("preview.bin", b"original")

    class Interrupted(io.BytesIO):
        def read(self, size):
            if chunk := super().read(size):
                return chunk
            if would_block:
                return None
            raise OSError("interrupted")

    stream = Interrupted(prefix)
    with pytest.raises(OSError, match="would block" if would_block else "interrupted"):
        manager.save_scratch_file(target.name, stream)
    assert target.read_bytes() == b"original" and stream.tell() == 0
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize(("content", "expected"), [
    (b"", b""), ("", b""), (io.BytesIO(), b""), (io.StringIO(), b""),
    (io.StringIO("caf\u00e9"), b"caf\xc3\xa9"),
])
def test_scratch_publishes_complete_payload_including_empty(tmp_path, content, expected):
    manager = DatasetManager(tmp_path / "scratch", tmp_path / "output")
    target = manager.save_scratch_file("preview.bin", b"original")
    assert manager.save_scratch_file(target.name, content) == target
    assert target.read_bytes() == expected


def test_scratch_short_write_preserves_target(tmp_path, monkeypatch):
    manager = DatasetManager(tmp_path / "scratch", tmp_path / "output")
    target = manager.save_scratch_file("preview.bin", b"original")
    new_file = tempfile.NamedTemporaryFile

    def short_writer(**kwargs):
        handle = new_file(**kwargs)
        write = handle.write
        handle.write = lambda data: write(data[:-1])
        return handle

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", short_writer)
    with pytest.raises(OSError, match="Incomplete scratch file write"):
        manager.save_scratch_file(target.name, b"new content")
    assert target.read_bytes() == b"original"
    assert list(target.parent.iterdir()) == [target]


def test_scratch_replace_failure_preserves_target(tmp_path, monkeypatch):
    manager = DatasetManager(tmp_path / "scratch", tmp_path / "output")
    target = manager.save_scratch_file("preview.bin", b"original")

    def failed_replace(staged, destination):
        assert staged.parent == target.parent and destination == target
        assert staged.read_bytes() == b"new content"
        raise PermissionError("replace denied")

    monkeypatch.setattr("core.dataset_manager.os.replace", failed_replace)
    with pytest.raises(PermissionError, match="replace denied"):
        manager.save_scratch_file(target.name, b"new content")
    assert target.read_bytes() == b"original"
    assert list(target.parent.iterdir()) == [target]
