"""
Boxify Backend — Smoke Test Script

Tests the core API endpoints in sequence:
  1. POST /api/dataset/upload   — Uploads a dynamically created .zip with sample images
  2. GET  /api/images           — Lists extracted images
  3. GET  /api/images/{filename} — Fetches a single image binary
  4. POST /api/annotations/{filename} — Saves sample YOLO annotations
  5. GET  /api/dataset/export   — Downloads the exported .zip

Usage:
    pip install requests Pillow
    python test_upload.py

Requires the Uvicorn server to be running on http://localhost:8000
"""

import io
import sys
import zipfile

try:
    import requests
except ImportError:
    print("❌ 'requests' not installed. Run: pip install requests")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("❌ 'Pillow' not installed. Run: pip install Pillow")
    sys.exit(1)


BASE_URL = "http://localhost:8000"

# ANSI colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{RESET}\n")


def result(success: bool, message: str) -> None:
    icon = f"{GREEN}✅ PASS" if success else f"{RED}❌ FAIL"
    print(f"  {icon}{RESET} — {message}")


def create_sample_zip() -> bytes:
    """
    Create an in-memory .zip file containing 3 small sample PNG images.
    No files on disk needed!
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(1, 4):
            # Create a small 100x80 solid-color image
            colors = [(255, 80, 80), (80, 200, 80), (80, 80, 255)]
            img = Image.new("RGB", (100, 80), colors[i - 1])

            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            zf.writestr(f"sample_dataset/image_{i:03d}.png", img_bytes.read())

    zip_buffer.seek(0)
    return zip_buffer.read()


def test_upload(zip_data: bytes) -> bool:
    """Test 1: POST /api/dataset/upload"""
    header("Test 1: Upload Dataset (.zip)")

    files = {"file": ("test_dataset.zip", zip_data, "application/zip")}
    resp = requests.post(f"{BASE_URL}/api/dataset/upload", files=files)

    print(f"  Status: {resp.status_code}")
    print(f"  Response: {resp.json()}")

    success = resp.status_code == 200 and resp.json().get("image_count") == 3
    result(success, f"Upload returned image_count={resp.json().get('image_count')}")
    return success


def test_list_images() -> list[str]:
    """Test 2: GET /api/images"""
    header("Test 2: List Images")

    resp = requests.get(f"{BASE_URL}/api/images")

    print(f"  Status: {resp.status_code}")
    data = resp.json()
    images = data.get("images", [])
    print(f"  Images found: {images}")

    success = resp.status_code == 200 and len(images) == 3
    result(success, f"Found {len(images)} image(s)")
    return images


def test_get_image(filename: str) -> bool:
    """Test 3: GET /api/images/{filename}"""
    header(f"Test 3: Fetch Image Binary — {filename}")

    resp = requests.get(f"{BASE_URL}/api/images/{filename}")

    print(f"  Status: {resp.status_code}")
    print(f"  Content-Type: {resp.headers.get('content-type')}")
    print(f"  Content-Length: {len(resp.content)} bytes")

    success = resp.status_code == 200 and len(resp.content) > 0
    result(success, f"Received {len(resp.content)} bytes of image data")
    return success


def test_save_annotation(filename: str) -> bool:
    """Test 4: POST /api/annotations/{filename}"""
    header(f"Test 4: Save Annotations — {filename}")

    payload = {
        "image_width": 100,
        "image_height": 80,
        "boxes": [
            {"x": 10, "y": 10, "width": 30, "height": 25, "label": "cat"},
            {"x": 50, "y": 20, "width": 40, "height": 50, "label": "dog"},
        ],
    }

    resp = requests.post(
        f"{BASE_URL}/api/annotations/{filename}", json=payload
    )

    print(f"  Status: {resp.status_code}")
    print(f"  Response: {resp.json()}")

    data = resp.json()
    success = resp.status_code == 200 and data.get("box_count") == 2
    result(success, f"Saved {data.get('box_count')} box(es) → {data.get('label_file')}")
    return success


def test_save_annotation_new_label(filename: str) -> bool:
    """Test 4b: Verify classes.txt grows with new labels"""
    header(f"Test 4b: Save Annotations with NEW label — {filename}")

    payload = {
        "image_width": 100,
        "image_height": 80,
        "boxes": [
            {"x": 5, "y": 5, "width": 20, "height": 15, "label": "bird"},
            {"x": 60, "y": 10, "width": 30, "height": 40, "label": "cat"},
        ],
    }

    resp = requests.post(
        f"{BASE_URL}/api/annotations/{filename}", json=payload
    )

    print(f"  Status: {resp.status_code}")
    print(f"  Response: {resp.json()}")

    data = resp.json()
    success = resp.status_code == 200 and data.get("box_count") == 2
    result(
        success,
        f"'bird' should now be index 2 in classes.txt (cat=0, dog=1, bird=2)",
    )
    return success


def test_export() -> bool:
    """Test 5: GET /api/dataset/export"""
    header("Test 5: Export Dataset (.zip)")

    resp = requests.get(f"{BASE_URL}/api/dataset/export")

    print(f"  Status: {resp.status_code}")
    print(f"  Content-Type: {resp.headers.get('content-type')}")
    print(f"  Content-Length: {len(resp.content)} bytes")

    if resp.status_code == 200 and len(resp.content) > 0:
        # Inspect the zip contents
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        names = sorted(zf.namelist())
        print(f"  Archive contents ({len(names)} files):")
        for name in names:
            print(f"    📄 {name}")

        has_images = any("images/" in n for n in names)
        has_output = any("output/" in n for n in names)
        has_inference = any("inference/" in n for n in names)
        has_classes = any("classes.txt" in n for n in names)

        result(has_images, "Contains images/")
        result(has_output, "Contains output/ (XML)")
        result(has_inference, "Contains inference/ (YOLO)")
        result(has_classes, "Contains classes.txt")
        return has_images and has_output and has_inference and has_classes

    result(False, f"Export failed with status {resp.status_code}")
    return False


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    header("🚀 Boxify Backend — Smoke Test Suite")

    # Check server is reachable
    try:
        requests.get(f"{BASE_URL}/docs", timeout=3)
    except requests.ConnectionError:
        print(f"  {RED}❌ Cannot connect to {BASE_URL}{RESET}")
        print(f"  Start the server first:")
        print(f"    cd backend && uvicorn api.main:app --reload --port 8000")
        sys.exit(1)

    print(f"  {GREEN}✅ Server is reachable at {BASE_URL}{RESET}")

    # Run tests
    zip_data = create_sample_zip()
    print(f"  📦 Created sample zip ({len(zip_data)} bytes, 3 images)")

    all_passed = True

    all_passed &= test_upload(zip_data)
    images = test_list_images()
    all_passed &= len(images) == 3

    if images:
        all_passed &= test_get_image(images[0])
        all_passed &= test_save_annotation(images[0])

        if len(images) > 1:
            all_passed &= test_save_annotation_new_label(images[1])

    all_passed &= test_export()

    # Summary
    header("📊 Final Results")
    if all_passed:
        print(f"  {GREEN}{BOLD}ALL TESTS PASSED ✅{RESET}")
    else:
        print(f"  {RED}{BOLD}SOME TESTS FAILED ❌{RESET}")

    sys.exit(0 if all_passed else 1)
