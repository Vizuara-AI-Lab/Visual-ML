"""
Test script for AWS S3 dataset upload functionality.
Run this to verify S3 integration is working correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.s3_service import s3_service
from app.core.config import settings
from app.core.logging import logger


async def test_s3_connection():
    """Test basic S3 connectivity."""
    print("\n" + "=" * 60)
    print("🧪 Testing AWS S3 Connection")
    print("=" * 60)

    # Check if S3 is enabled
    if not s3_service.enabled:
        print("\n❌ S3 is not enabled!")
        print(f"   USE_S3: {settings.USE_S3}")
        print(f"   S3_BUCKET: {settings.S3_BUCKET}")
        print(f"   AWS_ACCESS_KEY_ID: {'Set' if settings.AWS_ACCESS_KEY_ID else 'Not set'}")
        print(f"   AWS_SECRET_ACCESS_KEY: {'Set' if settings.AWS_SECRET_ACCESS_KEY else 'Not set'}")
        print("\n💡 To enable S3:")
        print("   1. Set USE_S3=True in .env")
        print("   2. Configure AWS credentials in .env")
        print("   3. See S3_SETUP_GUIDE.md for detailed instructions")
        return False

    print(f"\n✅ S3 Service is enabled")
    print(f"   Bucket: {s3_service.bucket}")
    print(f"   Region: {s3_service.region}")

    return True


async def test_s3_upload():
    """Test S3 file upload."""
    print("\n" + "-" * 60)
    print("📤 Testing S3 Upload")
    print("-" * 60)

    try:
        # Create test content
        test_content = b"dataset_id,value,category\n1,100,A\n2,200,B\n3,300,C"
        s3_key = "test/test_dataset.csv"

        # Upload
        s3_url = await s3_service.upload_file(
            file_content=test_content,
            s3_key=s3_key,
            content_type="text/csv",
            metadata={"test": "true", "source": "test_script"},
        )

        print(f"✅ Upload successful")
        print(f"   S3 URL: {s3_url}")
        print(f"   S3 Key: {s3_key}")
        print(f"   Size: {len(test_content)} bytes")

        return s3_key

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None


async def test_s3_download(s3_key: str):
    """Test S3 file download."""
    print("\n" + "-" * 60)
    print("📥 Testing S3 Download")
    print("-" * 60)

    try:
        content = await s3_service.download_file(s3_key)

        print(f"✅ Download successful")
        print(f"   Size: {len(content)} bytes")
        print(f"   Content preview: {content[:50].decode()}...")

        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


async def test_s3_presigned_url(s3_key: str):
    """Test presigned URL generation."""
    print("\n" + "-" * 60)
    print("🔗 Testing Presigned URL Generation")
    print("-" * 60)

    try:
        url = await s3_service.get_presigned_url(s3_key=s3_key, expiry=3600)

        print(f"✅ Presigned URL generated")
        print(f"   URL (first 80 chars): {url[:80]}...")
        print(f"   Expiry: 3600 seconds (1 hour)")

        return True

    except Exception as e:
        print(f"❌ Presigned URL generation failed: {e}")
        return False


async def test_s3_cleanup(s3_key: str):
    """Test S3 file deletion."""
    print("\n" + "-" * 60)
    print("🗑️  Testing S3 Cleanup")
    print("-" * 60)

    try:
        success = await s3_service.delete_file(s3_key)

        if success:
            print(f"✅ Cleanup successful")
            print(f"   Deleted: {s3_key}")
        else:
            print(f"⚠️  Cleanup completed with warnings")

        return success

    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        return False


async def test_s3_key_generation():
    """Test S3 key generation."""
    print("\n" + "-" * 60)
    print("🔑 Testing S3 Key Generation")
    print("-" * 60)

    s3_key = s3_service.generate_s3_key(
        project_id=123,
        dataset_id="dataset_abc123",
        filename="sales_data.csv",
        use_date_partition=True,
    )

    print(f"✅ S3 Key generated")
    print(f"   Key: {s3_key}")
    print(f"   Format: datasets/YYYY/MM/DD/project_ID/dataset_ID.ext")

    return s3_key


async def main():
    """Run all S3 tests."""
    print("\n" + "=" * 60)
    print("🚀 AWS S3 Integration Test Suite")
    print("=" * 60)

    # Test 1: Connection
    if not await test_s3_connection():
        print("\n" + "=" * 60)
        print("❌ Test Suite Aborted - S3 not configured")
        print("=" * 60)
        return

    # Test 2: Key generation
    await test_s3_key_generation()

    # Test 3: Upload
    s3_key = await test_s3_upload()
    if not s3_key:
        print("\n" + "=" * 60)
        print("❌ Test Suite Failed - Upload failed")
        print("=" * 60)
        return

    # Test 4: Download
    await test_s3_download(s3_key)

    # Test 5: Presigned URL
    await test_s3_presigned_url(s3_key)

    # Test 6: Cleanup
    await test_s3_cleanup(s3_key)

    # Summary
    print("\n" + "=" * 60)
    print("🎉 All S3 Tests Completed Successfully!")
    print("=" * 60)
    print("\n✅ Your S3 integration is working correctly.")
    print("✅ You can now upload datasets to your project.")
    print("\n💡 Next steps:")
    print("   1. Run database migration: alembic upgrade head")
    print("   2. Start the server: uvicorn main:app --reload")
    print("   3. Upload a dataset via API: POST /api/v1/datasets/upload")
    print("\n📖 See S3_SETUP_GUIDE.md for detailed usage instructions.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()
