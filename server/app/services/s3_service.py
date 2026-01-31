"""
AWS S3 Service - Production-ready S3 file storage utility.
Handles upload, download, presigned URLs, and lifecycle management.
"""

import boto3
from botocore.exceptions import ClientError, BotoCoreError
from boto3.s3.transfer import TransferConfig
from typing import Optional, BinaryIO
from datetime import datetime, timedelta
import io
from app.core.config import settings
from app.core.logging import logger


class S3Service:
    """Service for interacting with AWS S3."""

    def __init__(self):
        """Initialize S3 client with credentials from settings."""
        if settings.USE_S3 and settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            self.s3_client = boto3.client(
                "s3",
                region_name=settings.S3_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            )
            self.bucket = settings.S3_BUCKET
            self.region = settings.S3_REGION
            self.enabled = True
            logger.info(f"S3 Service initialized with bucket: {self.bucket}")
        else:
            self.s3_client = None
            self.bucket = None
            self.region = None
            self.enabled = False
            logger.warning("S3 Service disabled - missing credentials or USE_S3=False")

    async def upload_file(
        self,
        file_content: bytes,
        s3_key: str,
        content_type: str = "application/octet-stream",
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Upload file to S3.

        Args:
            file_content: File bytes to upload
            s3_key: S3 object key (path within bucket)
            content_type: MIME type
            metadata: Optional metadata dict

        Returns:
            S3 URL (s3://bucket/key)

        Raises:
            Exception: If upload fails
        """
        if not self.enabled:
            raise Exception("S3 service is not enabled. Check configuration.")

        try:
            # Prepare upload parameters
            upload_params = {
                "Bucket": self.bucket,
                "Key": s3_key,
                "Body": file_content,
                "ContentType": content_type,
            }

            # Add metadata if provided
            if metadata:
                upload_params["Metadata"] = {k: str(v) for k, v in metadata.items()}

            # Upload to S3
            self.s3_client.put_object(**upload_params)

            s3_url = f"s3://{self.bucket}/{s3_key}"
            logger.info(f"Successfully uploaded to S3: {s3_url}")
            return s3_url

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            logger.error(f"S3 upload failed [{error_code}]: {str(e)}")
            raise Exception(f"Failed to upload to S3: {error_code}")
        except BotoCoreError as e:
            logger.error(f"S3 BotoCore error: {str(e)}")
            raise Exception(f"S3 connection error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected S3 upload error: {str(e)}")
            raise

    async def upload_file_stream(
        self,
        file_obj: BinaryIO,
        s3_key: str,
        content_type: str = "application/octet-stream",
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Upload file to S3 using streaming (no memory buffering).
        Prevents buffer allocation errors for large files.

        Args:
            file_obj: File-like object to upload (e.g., UploadFile.file)
            s3_key: S3 object key (path within bucket)
            content_type: MIME type
            metadata: Optional metadata dict

        Returns:
            S3 URL (s3://bucket/key)

        Raises:
            Exception: If upload fails
        """
        if not self.enabled:
            raise Exception("S3 service is not enabled. Check configuration.")

        try:
            # Configure multipart upload for large files
            config = TransferConfig(
                multipart_threshold=10 * 1024 * 1024,  # 10MB
                max_concurrency=10,
                multipart_chunksize=10 * 1024 * 1024,  # 10MB chunks
                use_threads=True
            )

            # Prepare extra args
            extra_args = {
                "ContentType": content_type,
            }

            # Add metadata if provided
            if metadata:
                extra_args["Metadata"] = {k: str(v) for k, v in metadata.items()}

            # Upload using streaming (no memory buffering)
            self.s3_client.upload_fileobj(
                file_obj,
                self.bucket,
                s3_key,
                ExtraArgs=extra_args,
                Config=config
            )

            s3_url = f"s3://{self.bucket}/{s3_key}"
            logger.info(f"Successfully streamed upload to S3: {s3_url}")
            return s3_url

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            logger.error(f"S3 streaming upload failed [{error_code}]: {str(e)}")
            raise Exception(f"Failed to upload to S3: {error_code}")
        except BotoCoreError as e:
            logger.error(f"S3 BotoCore error: {str(e)}")
            raise Exception(f"S3 connection error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected S3 streaming upload error: {str(e)}")
            raise

    async def get_presigned_url(
        self, s3_key: str, expiry: int = 3600, operation: str = "get_object"
    ) -> str:
        """
        Generate presigned URL for temporary access to S3 object.

        Args:
            s3_key: S3 object key
            expiry: URL expiration time in seconds (default: 1 hour)
            operation: S3 operation ('get_object' or 'put_object')

        Returns:
            Presigned URL string

        Raises:
            Exception: If URL generation fails
        """
        if not self.enabled:
            raise Exception("S3 service is not enabled.")

        try:
            url = self.s3_client.generate_presigned_url(
                operation, Params={"Bucket": self.bucket, "Key": s3_key}, ExpiresIn=expiry
            )
            logger.info(f"Generated presigned URL for {s3_key} (expires in {expiry}s)")
            return url

        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {str(e)}")
            raise Exception(f"Failed to generate download URL: {str(e)}")

    async def download_file(self, s3_key: str) -> bytes:
        """
        Download file content from S3.

        Args:
            s3_key: S3 object key

        Returns:
            File content as bytes

        Raises:
            Exception: If download fails
        """
        if not self.enabled:
            raise Exception("S3 service is not enabled.")

        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            content = response["Body"].read()
            logger.info(f"Successfully downloaded from S3: {s3_key}")
            return content

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchKey":
                logger.error(f"S3 object not found: {s3_key}")
                raise Exception(f"File not found in S3: {s3_key}")
            logger.error(f"S3 download failed [{error_code}]: {str(e)}")
            raise Exception(f"Failed to download from S3: {error_code}")
        except Exception as e:
            logger.error(f"Unexpected S3 download error: {str(e)}")
            raise

    async def delete_file(self, s3_key: str) -> bool:
        """
        Delete file from S3.

        Args:
            s3_key: S3 object key

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.warning("S3 service is not enabled - cannot delete")
            return False

        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=s3_key)
            logger.info(f"Successfully deleted from S3: {s3_key}")
            return True

        except ClientError as e:
            logger.error(f"S3 deletion failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected S3 deletion error: {str(e)}")
            return False

    async def check_file_exists(self, s3_key: str) -> bool:
        """
        Check if file exists in S3.

        Args:
            s3_key: S3 object key

        Returns:
            True if exists, False otherwise
        """
        if not self.enabled:
            return False

        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError:
            return False

    async def get_file_metadata(self, s3_key: str) -> Optional[dict]:
        """
        Get file metadata from S3.

        Args:
            s3_key: S3 object key

        Returns:
            Metadata dict or None if not found
        """
        if not self.enabled:
            return None

        try:
            response = self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return {
                "content_length": response.get("ContentLength"),
                "content_type": response.get("ContentType"),
                "last_modified": response.get("LastModified"),
                "metadata": response.get("Metadata", {}),
            }
        except ClientError:
            return None

    def generate_s3_key(
        self, project_id: int, dataset_id: str, filename: str, use_date_partition: bool = True
    ) -> str:
        """
        Generate S3 key with proper structure.

        Args:
            project_id: Project ID
            dataset_id: Dataset ID
            filename: Original filename
            use_date_partition: Whether to use date-based partitioning

        Returns:
            S3 key string (e.g., "datasets/2026/01/23/project_123/dataset_abc.csv")
        """
        from pathlib import Path

        # Extract extension
        ext = Path(filename).suffix or ".csv"

        # Build key
        if use_date_partition:
            date_prefix = datetime.now().strftime("%Y/%m/%d")
            s3_key = f"datasets/{date_prefix}/project_{project_id}/{dataset_id}{ext}"
        else:
            s3_key = f"datasets/project_{project_id}/{dataset_id}{ext}"

        return s3_key


# Singleton instance
s3_service = S3Service()
