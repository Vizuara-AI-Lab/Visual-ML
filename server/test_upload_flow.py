"""
Simple test script to upload dataset directly to API endpoint.
This tests the complete upload flow: API -> Node -> S3 -> Database
"""

import asyncio
import sys
from pathlib import Path

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from main import app
from app.core.logging import logger

# Test client
client = TestClient(app)


def test_dataset_upload():
    """Test dataset upload through API endpoint."""
    print("\n" + "=" * 70)
    print("🧪 TESTING DATASET UPLOAD TO API")
    print("=" * 70)

    # Path to test CSV file
    csv_file = Path(__file__).parent.parent / "dataSet.csv"

    if not csv_file.exists():
        print(f"❌ Test file not found: {csv_file}")
        print(f"   Looking in: {csv_file.absolute()}")
        return False

    print(f"\n📄 Test file: {csv_file}")
    print(f"📏 File size: {csv_file.stat().st_size} bytes")

    # Test parameters
    project_id = 1

    print(f"\n🔧 Test Parameters:")
    print(f"   Project ID: {project_id}")
    print(f"   Filename: {csv_file.name}")

    # Upload file (NO authentication required now)
    print(f"\n🚀 Uploading to: POST /api/v1/datasets/upload?project_id={project_id}")

    with open(csv_file, "rb") as f:
        response = client.post(
            f"/api/v1/datasets/upload?project_id={project_id}",
            files={"file": (csv_file.name, f, "text/csv")},
        )

    print(f"\n📊 Response Status: {response.status_code}")
    print(f"📦 Response Body:")

    if response.status_code == 201:
        data = response.json()
        print(f"   ✅ Success: {data.get('message')}")
        print(f"\n📁 Dataset Info:")
        dataset = data.get("dataset", {})
        print(f"   Dataset ID: {dataset.get('dataset_id')}")
        print(f"   Filename: {dataset.get('filename')}")
        print(f"   Storage: {dataset.get('storage_backend')}")
        print(f"   S3 Bucket: {dataset.get('s3_bucket')}")
        print(f"   S3 Key: {dataset.get('s3_key')}")
        print(f"   Rows: {dataset.get('n_rows')}")
        print(f"   Columns: {dataset.get('n_columns')}")
        print(f"   File Size: {dataset.get('file_size')} bytes")
        return True
    else:
        print(f"   ❌ Failed: {response.text}")
        return False


def main():
    """Main test runner."""
    print("\n" + "=" * 70)
    print("🚀 DATASET UPLOAD TEST SCRIPT")
    print("=" * 70)

    success = test_dataset_upload()

    if success:
        print("\n" + "=" * 70)
        print("✅ TEST PASSED - Dataset uploaded successfully!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)


if __name__ == "__main__":
    main()

    # Run the test
    test_dataset_upload()

    print("\n" + "=" * 70)
    print("✨ TEST COMPLETE")
    print("=" * 70)
    print("\n💡 Next Steps:")
    print("   1. Start backend: cd server && uvicorn main:app --reload")
    print("   2. Start frontend: cd client && npm run dev")
    print("   3. Login to the app in browser")
    print("   4. Go to playground for a project")
    print("   5. Drag 'Upload Dataset' node")
    print("   6. Click node and select dataSet.csv")
    print("   7. Watch BOTH browser console AND backend terminal")
    print("")


if __name__ == "__main__":
    main()
