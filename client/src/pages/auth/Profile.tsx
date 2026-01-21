import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router";
import axiosInstance from "../../lib/axios";

interface UserProfile {
  id: number;
  emailId: string;
  collegeOrSchool?: string;
  contactNo?: string;
  recentProject?: string;
  profilePic?: string;
  isPremium: boolean;
  isActive: boolean;
}

const Profile: React.FC = () => {
  const navigate = useNavigate();
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [formData, setFormData] = useState({
    collegeOrSchool: "",
    contactNo: "",
    recentProject: "",
    profilePic: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  useEffect(() => {
    loadProfile();
  }, []);

  const loadProfile = async () => {
    try {
      const response = await axiosInstance.get("/auth/student/me");
      setProfile(response.data);
      setFormData({
        collegeOrSchool: response.data.collegeOrSchool || "",
        contactNo: response.data.contactNo || "",
        recentProject: response.data.recentProject || "",
        profilePic: response.data.profilePic || "",
      });
    } catch (err) {
      console.error("Failed to load profile:", err);
      setError("Failed to load profile");
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setSuccess("");

    try {
      const response = await axiosInstance.patch("/auth/student/me", formData);
      setProfile(response.data);
      setSuccess("Profile updated successfully!");
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to update profile");
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  if (!profile) {
    return <div className="p-4">Loading...</div>;
  }

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-2xl mx-auto bg-white p-6 rounded shadow">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold">My Profile</h1>
          <button
            onClick={() => navigate("/dashboard")}
            className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300"
          >
            Back to Dashboard
          </button>
        </div>

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

        <div className="mb-6 p-4 bg-gray-100 rounded">
          <p>
            <strong>Email:</strong> {profile.emailId}
          </p>
          <p>
            <strong>Status:</strong> {profile.isPremium ? "Premium" : "Free"}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block mb-1 font-medium">College/School</label>
            <input
              type="text"
              name="collegeOrSchool"
              value={formData.collegeOrSchool}
              onChange={handleChange}
              className="w-full px-3 py-2 border rounded"
              placeholder="Enter your college or school"
            />
          </div>

          <div>
            <label className="block mb-1 font-medium">Contact Number</label>
            <input
              type="text"
              name="contactNo"
              value={formData.contactNo}
              onChange={handleChange}
              className="w-full px-3 py-2 border rounded"
              placeholder="Enter your contact number"
            />
          </div>

          <div>
            <label className="block mb-1 font-medium">Recent Project</label>
            <textarea
              name="recentProject"
              value={formData.recentProject}
              onChange={handleChange}
              className="w-full px-3 py-2 border rounded"
              rows={4}
              placeholder="Describe your recent project"
            />
          </div>

          <div>
            <label className="block mb-1 font-medium">
              Profile Picture URL
            </label>
            <input
              type="url"
              name="profilePic"
              value={formData.profilePic}
              onChange={handleChange}
              className="w-full px-3 py-2 border rounded"
              placeholder="Enter profile picture URL"
            />
          </div>

          <div className="flex gap-4">
            <button
              type="submit"
              disabled={loading}
              className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
            >
              {loading ? "Updating..." : "Update Profile"}
            </button>
            <button
              type="button"
              onClick={() => navigate("/auth/change-password")}
              className="px-6 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
            >
              Change Password
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Profile;
