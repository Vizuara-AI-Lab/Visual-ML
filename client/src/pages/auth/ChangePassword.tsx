import React, { useState } from "react";
import { useNavigate } from "react-router";
import axiosInstance from "../../lib/axios";

const ChangePassword: React.FC = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    oldPassword: "",
    newPassword: "",
    confirmPassword: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    if (formData.newPassword !== formData.confirmPassword) {
      setError("New passwords do not match");
      return;
    }

    if (formData.newPassword.length < 8) {
      setError("Password must be at least 8 characters");
      return;
    }

    setLoading(true);

    try {
      await axiosInstance.post("/auth/student/change-password", {
        oldPassword: formData.oldPassword,
        newPassword: formData.newPassword,
      });
      setSuccess("Password changed successfully!");
      setTimeout(() => navigate("/profile"), 2000);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to change password");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4 flex items-center justify-center">
      <div className="max-w-md w-full bg-white p-6 rounded shadow">
        <h1 className="text-2xl font-bold mb-6">Change Password</h1>

        {error && (
          <div className="mb-4 p-3 bg-red-100 text-red-800 rounded">
            {error}
          </div>
        )}
        {success && (
          <div className="mb-4 p-3 bg-green-100 text-green-800 rounded">
            {success}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block mb-1 font-medium">Current Password</label>
            <input
              type="password"
              value={formData.oldPassword}
              onChange={(e) =>
                setFormData({ ...formData, oldPassword: e.target.value })
              }
              className="w-full px-3 py-2 border rounded"
              required
            />
          </div>

          <div>
            <label className="block mb-1 font-medium">New Password</label>
            <input
              type="password"
              value={formData.newPassword}
              onChange={(e) =>
                setFormData({ ...formData, newPassword: e.target.value })
              }
              className="w-full px-3 py-2 border rounded"
              required
            />
          </div>

          <div>
            <label className="block mb-1 font-medium">
              Confirm New Password
            </label>
            <input
              type="password"
              value={formData.confirmPassword}
              onChange={(e) =>
                setFormData({ ...formData, confirmPassword: e.target.value })
              }
              className="w-full px-3 py-2 border rounded"
              required
            />
          </div>

          <div className="flex gap-4">
            <button
              type="submit"
              disabled={loading}
              className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
            >
              {loading ? "Changing..." : "Change Password"}
            </button>
            <button
              type="button"
              onClick={() => navigate("/profile")}
              className="px-6 py-2 bg-gray-300 rounded hover:bg-gray-400"
            >
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ChangePassword;
