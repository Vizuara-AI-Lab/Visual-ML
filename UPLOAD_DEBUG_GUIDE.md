# 🐛 Dataset Upload Debug Guide

## Changes Made

### Frontend Logging Added

#### 1. **useUploadDataset Hook** (`client/src/hooks/mutations/useUploadDataset.ts`)

- ✅ Logs when upload starts
- ✅ Logs file name, size, and project ID
- ✅ Logs FormData creation
- ✅ Logs request URL
- ✅ Logs success/failure
- ✅ Added `onError` handler

#### 2. **NodeConfigPanel Component** (`client/src/components/playground/NodeConfigPanel.tsx`)

- ✅ Logs on component mount
- ✅ Logs project ID from URL params
- ✅ Logs when file input changes
- ✅ Logs selected file
- ✅ Logs upload start
- ✅ Logs upload completion
- ✅ Logs detailed errors
- ✅ Logs when upload is cancelled (no file/projectId)

#### 3. **Axios Instance** (`client/src/lib/axios.ts`)

- ✅ Request interceptor logs ALL outgoing requests
- ✅ Response interceptor logs ALL responses
- ✅ Logs full URL, method, headers, data

### Backend Logging Added

#### **Dataset Upload Endpoint** (`server/app/api/v1/datasets.py`)

- ✅ Logs when request received
- ✅ Logs project ID and student ID
- ✅ Logs file name and content type
- ✅ Logs file size in bytes and MB
- ✅ Logs validation steps
- ✅ Logs upload node creation
- ✅ Logs storage backend
- ✅ Logs upload execution
- ✅ Logs success with dataset ID

## How to Test

### 1. **Start Backend Server**

```bash
cd server
uvicorn main:app --reload
```

### 2. **Start Frontend Dev Server**

```bash
cd client
npm run dev
```

### 3. **Open Browser Console**

- Press F12 to open DevTools
- Go to Console tab
- Clear console (Ctrl+L)

### 4. **Test Upload Flow**

1. **Navigate to Playground:**
   - Go to `/playground/{project_id}`
   - Check console for: `[NodeConfigPanel] Project ID from params: {id}`

2. **Drag Upload Dataset Node:**
   - Drag "Upload File" node to canvas
   - Click on the node

3. **Click Upload Dataset Node:**
   - Configuration panel should open
   - Check console for:
     ```
     [NodeConfigPanel] Component mounted
     [NodeConfigPanel] Project ID from params: 123
     [NodeConfigPanel] Node type: upload_file
     ```

4. **Select a CSV File:**
   - Click "Choose File" or drag & drop
   - Check console for:
     ```
     [NodeConfigPanel] File input changed
     [NodeConfigPanel] Selected file: test.csv
     [NodeConfigPanel] Project ID: 123
     [NodeConfigPanel] Starting upload process...
     [NodeConfigPanel] Calling uploadDataset.mutateAsync...
     ```

5. **Monitor Upload:**
   - Check console for:
     ```
     [useUploadDataset] Starting upload...
     [useUploadDataset] File: test.csv 2048 bytes
     [useUploadDataset] Project ID: 123
     [useUploadDataset] FormData created, sending request...
     [useUploadDataset] URL: /datasets/upload?project_id=123
     [axios] Request: {...}
     ```

6. **Check Backend Logs:**
   - In server terminal, you should see:
     ```
     📤 Dataset upload request received - Project: 123, Student: 1
     📄 File: test.csv, Content-Type: text/csv
     ✅ File content read - Size: 2048 bytes (0.00 MB)
     📝 Validating file: test.csv
     🔧 Upload node created with storage backend: s3
     📦 Upload input prepared
     🚀 Executing upload node...
     ✅ Dataset uploaded successfully - ID: dataset_xyz, Project: 123, Student: 1, Storage: s3
     ```

7. **Check Success Response:**
   - In browser console:
     ```
     [axios] Response: {status: 201, data: {...}}
     [useUploadDataset] Upload successful!
     [useUploadDataset] Response: {success: true, dataset: {...}}
     [NodeConfigPanel] Upload completed successfully!
     ```

## Expected Console Output (Complete Flow)

### ✅ **Success Case:**

```
[NodeConfigPanel] Component mounted
[NodeConfigPanel] Project ID from params: 123
[NodeConfigPanel] File input changed
[NodeConfigPanel] Selected file: sales_data.csv
[NodeConfigPanel] Project ID: 123
[NodeConfigPanel] Starting upload process...
[NodeConfigPanel] Calling uploadDataset.mutateAsync...
[useUploadDataset] Starting upload...
[useUploadDataset] File: sales_data.csv 50000 bytes
[useUploadDataset] Project ID: 123
[useUploadDataset] FormData created, sending request...
[useUploadDataset] URL: /datasets/upload?project_id=123
[axios] Request: {method: 'POST', url: '/datasets/upload?project_id=123', ...}
[axios] Response: {status: 201, data: {success: true, ...}}
[useUploadDataset] Upload successful!
[useUploadDataset] Response: {success: true, dataset: {...}}
[useUploadDataset] onSuccess called
[NodeConfigPanel] Upload completed successfully!
[NodeConfigPanel] Upload state reset
```

### ❌ **If No Request Sent - Possible Issues:**

#### Issue 1: Project ID Missing

```
[NodeConfigPanel] Component mounted
[NodeConfigPanel] Project ID from params: undefined  ⚠️
[NodeConfigPanel] File input changed
[NodeConfigPanel] Upload cancelled - no file or project ID
[NodeConfigPanel] ProjectId: none  ⚠️
```

**Solution:** Check the URL - should be `/playground/{id}`

#### Issue 2: File Not Selected

```
[NodeConfigPanel] File input changed
[NodeConfigPanel] Selected file: undefined  ⚠️
[NodeConfigPanel] Upload cancelled - no file or project ID
```

**Solution:** Make sure file input has `accept=".csv"` and file is selected

#### Issue 3: Mutation Hook Not Initialized

```
[NodeConfigPanel] uploadDataset hook status: idle  ⚠️
```

**Solution:** Check if `useUploadDataset` hook is properly imported

## Troubleshooting

### No Console Logs at All

- Check if DevTools console is open
- Check if console is filtered - set to "All levels"
- Try refresh page (Ctrl+R)

### Logs Stop at "Calling uploadDataset.mutateAsync"

- Check Network tab for failed requests
- Look for CORS errors
- Check if backend server is running
- Verify `VITE_API_URL` in `.env`

### Backend Not Receiving Request

- Check Network tab - is request being sent?
- Check request URL - should be `http://localhost:8000/api/v1/datasets/upload?project_id=123`
- Check authentication - cookies should be sent
- Check CORS configuration

### Upload Fails with 401 Unauthorized

- User not logged in
- Check cookies in DevTools → Application → Cookies
- Re-login to get fresh tokens

### Upload Fails with 404 Not Found

- Check project exists
- Check project belongs to current student
- Verify `project_id` parameter

### S3 Upload Fails

- Check `.env` file has correct AWS credentials
- Run `python test_s3.py` to verify S3 connectivity
- Check S3 bucket permissions
- Check IAM user permissions

## API Endpoint Details

**URL:** `POST /api/v1/datasets/upload`

**Query Parameters:**

- `project_id` (required): Project ID to associate dataset with

**Body:**

- `file` (multipart/form-data): CSV file to upload

**Headers:**

- `Content-Type: multipart/form-data`
- Cookies: Authentication cookies

**Response (201 Created):**

```json
{
  "success": true,
  "message": "Dataset uploaded successfully",
  "dataset": {
    "dataset_id": "dataset_abc123",
    "filename": "sales_data.csv",
    "file_path": "projects/123/datasets/2026/01/24/dataset_abc123.csv",
    "storage_backend": "s3",
    "s3_bucket": "visual-ml-datasets-prod",
    "s3_key": "projects/123/datasets/2026/01/24/dataset_abc123.csv",
    "n_rows": 1000,
    "n_columns": 10,
    "columns": ["col1", "col2", ...],
    "dtypes": {"col1": "int64", ...},
    "memory_usage_mb": 1.95,
    "file_size": 50000,
    "preview": [{...}, {...}, ...]
  }
}
```

## Database Verification

After successful upload, check database:

```sql
-- Check if dataset was saved
SELECT * FROM datasets
WHERE project_id = 123
ORDER BY created_at DESC
LIMIT 1;

-- Check all datasets for student
SELECT dataset_id, filename, n_rows, n_columns, storage_backend, created_at
FROM datasets
WHERE user_id = 1
ORDER BY created_at DESC;
```

## S3 Verification

Check if file was uploaded to S3:

```bash
aws s3 ls s3://visual-ml-datasets-prod/projects/123/datasets/2026/01/24/ --recursive
```

Or use AWS Console:

1. Go to S3 Console
2. Open bucket: `visual-ml-datasets-prod`
3. Navigate to: `projects/123/datasets/2026/01/24/`
4. Look for `dataset_*.csv` files

---

**After testing, please share:**

1. ✅ All console logs from browser
2. ✅ Backend server logs
3. ✅ Network tab screenshot (if request is sent)
4. ✅ Any error messages
