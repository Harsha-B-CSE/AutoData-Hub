import io
import time
from pathlib import Path
import cv2
import numpy as np
import pytest

def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "worker_cores_allocated" in data

def test_config_presets(client):
    response = client.get("/api/v1/config/presets")
    assert response.status_code == 200
    data = response.json()
    assert "built_in" in data
    assert len(data["built_in"]) >= 2

def test_upload_and_preview_workflow(client):
    csv_content = (
        "id,name,age,salary\n"
        "1,Alice,25,50000\n"
        "2,Bob,,60000\n"
        "3,Charlie,35,\n"
        "1,Alice,25,50000\n"
    )

    # 1. Upload
    response = client.post(
        "/api/v1/upload",
        files={"file": ("test_employees.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    )
    assert response.status_code == 200
    upload_data = response.json()
    dataset_id = upload_data["dataset_id"]
    assert upload_data["type"] == "tabular"

    # 2. Preview
    preview_res = client.get(f"/api/v1/datasets/{dataset_id}/preview?page=1&limit=10")
    assert preview_res.status_code == 200
    preview_data = preview_res.json()
    assert preview_data["total_rows"] == 4
    assert "profile" in preview_data
    assert preview_data["profile"]["duplicate_rows"] == 1

    # 3. Dispatch processing job
    process_res = client.post(
        "/api/v1/process",
        json={
            "dataset_id": dataset_id,
            "config_id": "builtin-tabular-standard",
            "execution_mode": "batch"
        }
    )
    assert process_res.status_code == 200
    job_id = process_res.json()["job_id"]

    # 4. Wait for job to complete
    max_wait = 10
    completed = False
    for _ in range(max_wait):
        time.sleep(0.5)
        status_res = client.get(f"/api/v1/jobs/{job_id}")
        assert status_res.status_code == 200
        status_data = status_res.json()
        if status_data["status"] == "COMPLETED":
            completed = True
            break
        elif status_data["status"] == "FAILED":
            pytest.fail(f"Job failed: {status_data.get('error_message')}")

    assert completed, "Job did not complete in time"

    # 5. Check quality report
    quality_res = client.get(f"/api/v1/jobs/{job_id}/quality")
    assert quality_res.status_code == 200
    qr = quality_res.json()
    assert qr["post_score"] >= qr["pre_score"]

    # 6. Check diff
    diff_res = client.get(f"/api/v1/jobs/{job_id}/diff")
    assert diff_res.status_code == 200
    diff_data = diff_res.json()
    assert diff_data["type"] == "tabular"
    # Duplicates should have been removed: 4 rows -> 3 rows
    assert diff_data["total_after"] == 3

    # 7. Check PDF Export
    pdf_res = client.get(f"/api/v1/jobs/{job_id}/export?format=pdf")
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert len(pdf_res.content) > 1000

    # 8. Check CSV Export
    csv_res = client.get(f"/api/v1/jobs/{job_id}/export?format=csv")
    assert csv_res.status_code == 200
    assert "Alice" in csv_res.text

    # 9. Check Excel (.xlsx) Export
    xlsx_res = client.get(f"/api/v1/jobs/{job_id}/export?format=xlsx")
    assert xlsx_res.status_code == 200
    assert "spreadsheetml" in xlsx_res.headers["content-type"]
    assert len(xlsx_res.content) > 1000

    # 10. Check JSON Manifest Export
    json_res = client.get(f"/api/v1/jobs/{job_id}/export?format=json")
    assert json_res.status_code == 200
    manifest_data = json_res.json()
    assert manifest_data["engine"] == "tabular"
    assert "pre_score" in manifest_data
    assert "post_score" in manifest_data
    assert "parameters" in manifest_data

    # 11. Verify real performance benchmarks
    bench_res = client.get(f"/api/v1/jobs/{job_id}/benchmarks")
    assert bench_res.status_code == 200
    bench_data = bench_res.json()
    assert len(bench_data) >= 5
    for b in bench_data:
        assert b["latency_ms"] > 0
        assert "stage_name" in b

def test_vision_upload_and_process_workflow(client):
    import cv2
    import numpy as np

    # Create dummy PNG image bytes
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, (255, 255, 255), -1)
    is_success, buffer = cv2.imencode(".png", img)
    assert is_success

    # Upload single image
    res = client.post(
        "/api/v1/upload",
        files={"file": ("test_circle.png", io.BytesIO(buffer.tobytes()), "image/png")}
    )
    assert res.status_code == 200
    upload_data = res.json()
    assert upload_data["type"] == "vision"
    ds_id = upload_data["dataset_id"]

    # Preview
    preview_res = client.get(f"/api/v1/datasets/{ds_id}/preview")
    assert preview_res.status_code == 200
    p_data = preview_res.json()
    assert p_data["type"] == "vision"
    assert p_data["total_images"] == 1

    # Dispatch job
    proc_res = client.post(
        "/api/v1/process",
        json={
            "dataset_id": ds_id,
            "config_id": "builtin-vision-standard",
            "execution_mode": "batch"
        }
    )
    assert proc_res.status_code == 200
    job_id = proc_res.json()["job_id"]

    # Wait for completion
    completed = False
    for _ in range(15):
        time.sleep(0.5)
        st_res = client.get(f"/api/v1/jobs/{job_id}")
        if st_res.json()["status"] == "COMPLETED":
            completed = True
            break
        elif st_res.json()["status"] == "FAILED":
            pytest.fail(f"Vision job failed: {st_res.json().get('error_message')}")
    assert completed

    # Diff
    diff_res = client.get(f"/api/v1/jobs/{job_id}/diff")
    assert diff_res.status_code == 200
    assert diff_res.json()["type"] == "vision"

    # Export ZIP
    zip_res = client.get(f"/api/v1/jobs/{job_id}/export?format=zip")
    assert zip_res.status_code == 200
    assert zip_res.headers["content-type"] == "application/zip"

    # Export PNG
    png_res = client.get(f"/api/v1/jobs/{job_id}/export?format=png")
    assert png_res.status_code == 200
    assert png_res.headers["content-type"] == "image/png"
    assert len(png_res.content) > 0

    # Export JPEG
    jpg_res = client.get(f"/api/v1/jobs/{job_id}/export?format=jpg")
    assert jpg_res.status_code == 200
    assert jpg_res.headers["content-type"] == "image/jpeg"
    assert len(jpg_res.content) > 0

    # Dedicated single image download route
    dl_res = client.get(f"/api/v1/jobs/{job_id}/download-image?format=png")
    assert dl_res.status_code == 200
    assert dl_res.headers["content-type"] == "image/png"

def test_mysore_palace_e2e_color_preservation(client):
    """
    Verify that the synthetic golden night scene retains its warm colors after upload,
    processing, and download. The fixture is an illustration, not a photograph.
    """
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "Mysore_palace_illuminated.jpg"
    with fixture_path.open("rb") as f:
        upload_res = client.post("/api/v1/upload", files={"file": (fixture_path.name, f, "image/jpeg")})
    assert upload_res.status_code == 200
    ds_id = upload_res.json()["dataset_id"]

    # Dispatch with standard vision preset (Bilateral + CLAHE)
    proc_res = client.post(
        "/api/v1/process",
        json={
            "dataset_id": ds_id,
            "config_id": "builtin-vision-standard",
            "execution_mode": "batch"
        }
    )
    assert proc_res.status_code == 200
    job_id = proc_res.json()["job_id"]

    # Wait for completion
    completed = False
    for _ in range(30):
        time.sleep(0.5)
        st_res = client.get(f"/api/v1/jobs/{job_id}")
        if st_res.json()["status"] == "COMPLETED":
            completed = True
            break
        elif st_res.json()["status"] == "FAILED":
            pytest.fail(f"Mysore palace job failed: {st_res.json().get('error_message')}")
    assert completed, "Job did not complete in time"

    # Download processed image via API
    dl_res = client.get(f"/api/v1/jobs/{job_id}/download-image?format=png")
    assert dl_res.status_code == 200
    assert dl_res.headers["content-type"] == "image/png"

    # Decode bytes to OpenCV BGR array
    img_array = np.frombuffer(dl_res.content, dtype=np.uint8)
    proc_bgr = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    assert proc_bgr is not None
    assert len(proc_bgr.shape) == 3

    # Check channels: Red is channel 2, Blue is channel 0
    mean_b = proc_bgr[:, :, 0].mean()
    mean_g = proc_bgr[:, :, 1].mean()
    mean_r = proc_bgr[:, :, 2].mean()

    assert mean_r > mean_b, f"Color inversion detected! Red ({mean_r}) is not greater than Blue ({mean_b})"
    assert mean_r > mean_g, f"Red ({mean_r}) should be higher than Green ({mean_g}) for warm golden lighting"

def test_dataset_specific_export_enforcement(client):
    """
    Verifies that export endpoints strictly enforce format validity per dataset type:
    - Tabular datasets allow CSV, XLSX, PDF, JSON; strictly reject PNG, JPG, ZIP.
    - Image datasets allow PNG, JPG, ZIP, PDF, JSON; strictly reject CSV, XLSX.
    """
    # 1. Tabular workflow
    csv_bytes = b"id,name,val\n1,Alpha,10\n2,Beta,20\n"
    up_tab = client.post("/api/v1/upload", files={"file": ("test_export_tab.csv", io.BytesIO(csv_bytes), "text/csv")})
    assert up_tab.status_code == 200
    tab_ds_id = up_tab.json()["dataset_id"]

    proc_tab = client.post("/api/v1/process", json={"dataset_id": tab_ds_id, "config_id": "builtin-tabular-standard"})
    assert proc_tab.status_code == 200
    tab_job_id = proc_tab.json()["job_id"]

    # Wait for completion
    for _ in range(20):
        time.sleep(0.3)
        if client.get(f"/api/v1/jobs/{tab_job_id}").json()["status"] == "COMPLETED":
            break

    # Tabular valid formats
    assert client.get(f"/api/v1/jobs/{tab_job_id}/export?format=csv").status_code == 200
    assert client.get(f"/api/v1/jobs/{tab_job_id}/export?format=xlsx").status_code == 200
    assert client.get(f"/api/v1/jobs/{tab_job_id}/export?format=pdf").status_code == 200
    assert client.get(f"/api/v1/jobs/{tab_job_id}/export?format=json").status_code == 200

    # Tabular invalid image formats must return 400 Bad Request
    png_res = client.get(f"/api/v1/jobs/{tab_job_id}/export?format=png")
    assert png_res.status_code == 400
    assert "not supported for tabular" in png_res.json()["detail"].lower()

    jpg_res = client.get(f"/api/v1/jobs/{tab_job_id}/export?format=jpg")
    assert jpg_res.status_code == 400
    assert "not supported for tabular" in jpg_res.json()["detail"].lower()

    zip_res = client.get(f"/api/v1/jobs/{tab_job_id}/export?format=zip")
    assert zip_res.status_code == 400
    assert "not supported for tabular" in zip_res.json()["detail"].lower()

    dl_img_res = client.get(f"/api/v1/jobs/{tab_job_id}/download-image?format=png")
    assert dl_img_res.status_code == 400
    assert "visual datasets" in dl_img_res.json()["detail"].lower()

    # 2. Vision workflow
    syn_img = np.zeros((60, 60, 3), dtype=np.uint8)
    syn_img[10:50, 10:50] = [50, 100, 200]
    _, buf = cv2.imencode(".png", syn_img)
    up_vis = client.post("/api/v1/upload", files={"file": ("test_export_vis.png", io.BytesIO(buf.tobytes()), "image/png")})
    assert up_vis.status_code == 200
    vis_ds_id = up_vis.json()["dataset_id"]

    proc_vis = client.post("/api/v1/process", json={"dataset_id": vis_ds_id, "config_id": "builtin-vision-standard"})
    assert proc_vis.status_code == 200
    vis_job_id = proc_vis.json()["job_id"]

    for _ in range(20):
        time.sleep(0.3)
        if client.get(f"/api/v1/jobs/{vis_job_id}").json()["status"] == "COMPLETED":
            break

    # Vision valid formats
    assert client.get(f"/api/v1/jobs/{vis_job_id}/export?format=png").status_code == 200
    assert client.get(f"/api/v1/jobs/{vis_job_id}/export?format=jpg").status_code == 200
    assert client.get(f"/api/v1/jobs/{vis_job_id}/export?format=zip").status_code == 200
    assert client.get(f"/api/v1/jobs/{vis_job_id}/download-image?format=png").status_code == 200

    # Vision invalid tabular formats must return 400 Bad Request
    csv_vis = client.get(f"/api/v1/jobs/{vis_job_id}/export?format=csv")
    assert csv_vis.status_code == 400
    assert "not supported for image" in csv_vis.json()["detail"].lower()

    xlsx_vis = client.get(f"/api/v1/jobs/{vis_job_id}/export?format=xlsx")
    assert xlsx_vis.status_code == 400
    assert "not supported for image" in xlsx_vis.json()["detail"].lower()

    # Verify diff response includes adaptive enhancement intelligence
    diff_res = client.get(f"/api/v1/jobs/{vis_job_id}/diff")
    assert diff_res.status_code == 200
    comps = diff_res.json().get("comparisons", [])
    assert len(comps) > 0
    assert "enhancements" in comps[0]
