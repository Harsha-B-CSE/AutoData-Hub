"""Download and checksum-verify optional ONNX image-restoration weights."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen


MODEL_FILES = {
    "nafnet": [
        {
            "filename": "nafnet.onnx",
            "url": "https://github.com/rock1110/nafnet-onnx/releases/download/v1.0.0/nafnet-width32-sidd-fp32.onnx",
            "sha256": "06b6558d8890db9010c1fc524e26552d01d43fd3ba0c935507e5e9fc33876d76",
            "license": "unspecified third-party ONNX export",
        }
    ],
    "restormer": [
        {
            "filename": "restormer.onnx",
            "url": "https://github.com/TechFlipsi/FlipsiColor-Models/releases/download/v0.2/RestormerLight.onnx",
            "sha256": "63759916b606b0dace088baf983b6379bacc441e79411cb1af915131eb763653",
            "license": "unspecified third-party ONNX export",
        }
    ],
    "real-esrgan": [
        {
            "filename": "real_esrgan_x4.onnx",
            "url": "https://huggingface.co/TheGuy444/Real-ESRGAN-ONNX/resolve/main/onnx/model.onnx",
            "sha256": "2d1810f8b75c6a899487decf933697241eafbe6327d7b723674ba6df73a08fda",
            "license": "BSD-3-Clause",
        },
        {
            "filename": "model.data",
            "url": "https://huggingface.co/TheGuy444/Real-ESRGAN-ONNX/resolve/main/onnx/model.data",
            "sha256": "512d0ec9940c2e9d85d27f2952f12a0b77b7841dc22df4ce9f3ea458bc98f37f",
            "license": "BSD-3-Clause",
        },
    ],
    "swinir": [
        {
            "filename": "swinir.onnx",
            "url": "https://huggingface.co/rocca/swin-ir-onnx/resolve/main/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx",
            "sha256": "50a7b6fcaad6c4f342c5e1d059916a3e5d48e89d1d311270a476a7ce6a2ea09f",
            "license": "Apache-2.0",
        }
    ],
    "ultrasharp-v2": [
        {
            "filename": "ultrasharp_v2_x4.onnx",
            "url": "https://huggingface.co/huggingworld/onnx-image-models/resolve/main/4x-UltraSharpV2_fp32_op17.onnx",
            "sha256": "6c0201e3403745f39a9aa5273c50ed084cf6c4af5a71a4654c30252cf19bf0a5",
            "license": "CC-BY-NC-SA-4.0 (non-commercial)",
        }
    ],
    "dat-nomos8k": [
        {
            "filename": "dat_nomos8k_x4.onnx",
            "url": "https://huggingface.co/huggingworld/onnx-image-models/resolve/main/4xNomos8kDAT.onnx",
            "sha256": "e5c10de92a14544764ca4e4dc0269f7de3cd2e4975b1d149ad687fd195eb9de1",
            "license": "MIT hub card; community checkpoint provenance",
        }
    ],
    "realplksr": [
        {
            "filename": "realplksr_x4.onnx",
            "url": "https://huggingface.co/darktable-org/upscale-realplksr-onnx/resolve/0bb636eeeed5fffd256571c3cc440d9fdf62467e/onnx/model_x4.onnx",
            "sha256": "8ce692652494e2e5c31e1f7c7ab61eb084e9a450cde727a374f5d6746cfe054b",
            "license": "MIT",
        }
    ],
}

DEFAULT_MODEL_IDS = [
    "nafnet",
    "restormer",
    "real-esrgan",
    "swinir",
    "realplksr",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified_file(entry: dict, destination: Path, force: bool = False) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not force:
        if sha256_file(destination) == entry["sha256"]:
            return "verified-existing"

    request = Request(
        entry["url"],
        headers={"User-Agent": "AutoData-Hub-Model-Setup/1.0"},
    )
    with tempfile.NamedTemporaryFile(
        prefix=f"{destination.name}.",
        suffix=".download",
        dir=destination.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, temporary, length=1024 * 1024)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    actual_hash = sha256_file(temporary_path)
    if actual_hash != entry["sha256"]:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {destination.name}: {actual_hash}"
        )
    temporary_path.replace(destination)
    return "downloaded"


def provision_models(
    model_ids: list[str],
    model_directory: Path,
    force: bool = False,
) -> list[dict]:
    results = []
    for model_id in model_ids:
        for entry in MODEL_FILES[model_id]:
            destination = model_directory / entry["filename"]
            status = download_verified_file(entry, destination, force=force)
            results.append({
                "model": model_id,
                "file": str(destination),
                "status": status,
                "license": entry["license"],
            })
    return results


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download checksum-verified AutoData-Hub ONNX models."
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(MODEL_FILES),
        help="Provision only this model; repeat for multiple models.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also provision experimental and non-commercial model entries.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=project_root / "models" / "image_restoration",
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    if arguments.all and arguments.model:
        parser.error("Use either --all or one/more --model options, not both.")
    selected = (
        list(MODEL_FILES)
        if arguments.all
        else arguments.model or DEFAULT_MODEL_IDS
    )
    results = provision_models(selected, arguments.model_dir, force=arguments.force)
    for result in results:
        print(
            f"[{result['status']}] {result['model']}: {result['file']} "
            f"({result['license']})"
        )


if __name__ == "__main__":
    main()
