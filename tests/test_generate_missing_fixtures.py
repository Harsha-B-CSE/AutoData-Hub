import runpy
import shutil
from pathlib import Path

import cv2
import numpy as np


def test_generate_synthetic_fixture_from_another_cwd(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    script = repo / "scripts" / "generate_missing_fixtures.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts" / script.name, script)
    monkeypatch.chdir(tmp_path)

    runpy.run_path(str(script), run_name="__main__")
    fixture = repo / "tests" / "fixtures" / "Mysore_palace_illuminated.jpg"
    image = cv2.imread(str(fixture), cv2.IMREAD_UNCHANGED)
    assert image is not None and image.shape == (480, 640, 3)
    np.testing.assert_allclose(image[240, 320], (0, 215, 255), atol=4)
    assert image[240, 320, 2] > image[240, 200, 2] > image[240, 0, 2]

    original = fixture.read_bytes()
    fixture.unlink()
    runpy.run_path(str(script), run_name="__main__")
    assert fixture.read_bytes() == original

    fixture.write_bytes(b"existing fixture")
    runpy.run_path(str(script), run_name="__main__")
    assert fixture.read_bytes() == b"existing fixture"
