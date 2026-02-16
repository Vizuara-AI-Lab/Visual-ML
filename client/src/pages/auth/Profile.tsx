import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router";
import {
  User,
  Mail,
  School,
  Phone,
  FileText,
  Camera,
  ArrowLeft,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Crown,
  X,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import axiosInstance from "../../lib/axios";
import Navbar from "../../landingpage/Navbar";
import { MentorSettings } from "../../components/MentorSettings";

interface UserProfile {
  id: number;
  emailId: string;
  fullName?: string;
  collegeOrSchool?: string;
  contactNo?: string;
  recentProject?: string;
  profilePic?: string;
  isPremium: boolean;
  isActive: boolean;
}

const Profile: React.FC = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [formData, setFormData] = useState({
    collegeOrSchool: "",
    contactNo: "",
    recentProject: "",
    profilePic: "",
  });
  const [loading, setLoading] = useState(false);
  const [uploadingImage, setUploadingImage] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);

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
      if (response.data.profilePic) {
        setPreviewImage(response.data.profilePic);
      }
    } catch (err) {
      console.error("Failed to load profile:", err);
      setError("Failed to load profile");
    }
  };

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith("image/")) {
      setError("Please select an image file");
      return;
    }

    // Validate file size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      setError("Image size must be less than 5MB");
      return;
    }

    setUploadingImage(true);
    setError("");
    setUploadProgress(0);

    try {
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewImage(reader.result as string);
      };
      reader.readAsDataURL(file);

      // Upload to server
      const formData = new FormData();
      formData.append("file", file);

      const response = await axiosInstance.post(
        "/auth/student/upload-profile-pic",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          onUploadProgress: (progressEvent) => {
            if (progressEvent.total) {
              const progress = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total,
              );
              setUploadProgress(progress);
            }
          },
        },
      );

      // Update form data with new URL
      setFormData((prev) => ({
        ...prev,
        profilePic: response.data.profilePicUrl,
      }));

      setSuccess("Profile picture uploaded successfully!");
      setTimeout(() => setSuccess(""), 3000);
    } catch (err: any) {
      console.error("Upload error:", err);
      setError(
        err.response?.data?.detail ||
          "Failed to upload image. Please try again.",
      );
      setPreviewImage(formData.profilePic || null);
    } finally {
      setUploadingImage(false);
      setUploadProgress(0);
    }
  };

  const handleRemoveImage = () => {
    setPreviewImage(null);
    setFormData((prev) => ({ ...prev, profilePic: "" }));
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
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
      setTimeout(() => setSuccess(""), 3000);
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
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="flex items-center gap-3">
          <Loader2 className="w-6 h-6 animate-spin text-slate-600" />
          <span className="text-slate-600">Loading profile...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 pt-20">
      <Navbar />
      {/* Background Pattern */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#f0f0f0_1px,transparent_1px),linear-gradient(to_bottom,#f0f0f0_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-40" />
      </div>

      <div className="relative max-w-4xl mx-auto px-4 py-8 lg:py-12">
        {/* Header */}
        <div className="mb-8">
          <button
            onClick={() => navigate("/dashboard")}
            className="flex items-center gap-2 text-slate-600 hover:text-slate-900 transition-colors mb-6 group"
          >
            <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
            <span className="text-sm font-medium">Back to Dashboard</span>
          </button>

          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl lg:text-4xl font-bold bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 bg-clip-text text-transparent">
                My Profile
              </h1>
              <p className="text-slate-600 mt-2">
                Manage your account settings and preferences
              </p>
            </div>

            {profile.isPremium && (
              <div className="flex items-center gap-2 px-4 py-2 bg-linear-to-r from-amber-400 to-yellow-500 text-white rounded-xl shadow-lg">
                <Crown className="w-5 h-5" />
                <span className="font-semibold">Premium</span>
              </div>
            )}
          </div>
        </div>

        {/* Alerts */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl flex items-start gap-3"
            >
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="text-sm font-medium text-red-900">Error</p>
                <p className="text-sm text-red-700">{error}</p>
              </div>
              <button
                onClick={() => setError("")}
                className="text-red-400 hover:text-red-600"
              >
                <X className="w-4 h-4" />
              </button>
            </motion.div>
          )}

          {success && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mb-6 p-4 bg-green-50 border border-green-200 rounded-xl flex items-start gap-3"
            >
              <CheckCircle2 className="w-5 h-5 text-green-600 shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="text-sm font-medium text-green-900">Success</p>
                <p className="text-sm text-green-700">{success}</p>
              </div>
              <button
                onClick={() => setSuccess("")}
                className="text-green-400 hover:text-green-600"
              >
                <X className="w-4 h-4" />
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left Column - Profile Picture & Info */}
          <div className="lg:col-span-1 space-y-6">
            {/* Profile Picture Card */}
            <div className="bg-white rounded-2xl shadow-xl border border-slate-200/60 p-6">
              <h2 className="text-lg font-semibold text-slate-900 mb-4">
                Profile Picture
              </h2>

              <div className="flex flex-col items-center">
                {/* Image Preview */}
                <div className="relative group">
                  <div className="w-32 h-32 rounded-full overflow-hidden bg-slate-100 border-4 border-white shadow-xl">
                    {previewImage ? (
                      <img
                        src={previewImage}
                        alt="Profile"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center bg-linear-to-br from-slate-900 to-slate-700">
                        <User className="w-16 h-16 text-white" />
                      </div>
                    )}
                  </div>

                  {/* Remove Button */}
                  {previewImage && !uploadingImage && (
                    <button
                      onClick={handleRemoveImage}
                      className="absolute -top-2 -right-2 w-8 h-8 bg-red-500 text-white rounded-full flex items-center justify-center shadow-lg hover:bg-red-600 transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}

                  {/* Upload Progress Overlay */}
                  {uploadingImage && (
                    <div className="absolute inset-0 bg-slate-900/50 rounded-full flex items-center justify-center">
                      <div className="text-center">
                        <Loader2 className="w-8 h-8 text-white animate-spin mx-auto mb-1" />
                        <span className="text-xs text-white font-medium">
                          {uploadProgress}%
                        </span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Upload Button */}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                  disabled={uploadingImage}
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={uploadingImage}
                  className="mt-4 w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-slate-900 text-white rounded-xl font-medium hover:bg-slate-800 transition-all shadow-lg shadow-slate-900/25 hover:shadow-xl hover:shadow-slate-900/30 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {uploadingImage ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Uploading...</span>
                    </>
                  ) : (
                    <>
                      <Camera className="w-4 h-4" />
                      <span>
                        {previewImage ? "Change Photo" : "Upload Photo"}
                      </span>
                    </>
                  )}
                </button>
                <p className="text-xs text-slate-500 mt-2 text-center">
                  Max size: 5MB. Supported: JPG, PNG, GIF
                </p>
              </div>
            </div>

            {/* Account Info Card */}
            <div className="bg-white rounded-2xl shadow-xl border border-slate-200/60 p-6">
              <h2 className="text-lg font-semibold text-slate-900 mb-4">
                Account Info
              </h2>
              <div className="space-y-3">
                <div className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg">
                  <Mail className="w-5 h-5 text-slate-400" />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-slate-500">Email</p>
                    <p className="text-sm font-medium text-slate-900 truncate">
                      {profile.emailId}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg">
                  <Crown className="w-5 h-5 text-slate-400" />
                  <div className="flex-1">
                    <p className="text-xs text-slate-500">Plan</p>
                    <p className="text-sm font-medium text-slate-900">
                      {profile.isPremium ? "Premium" : "Free"}
                    </p>
                  </div>
                </div>
              </div>

              <button
                onClick={() => navigate("/auth/change-password")}
                className="mt-4 w-full px-4 py-2.5 bg-slate-100 text-slate-900 rounded-xl font-medium hover:bg-slate-200 transition-all"
              >
                Change Password
              </button>
            </div>
          </div>

          {/* Right Column - Edit Form */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-2xl shadow-xl border border-slate-200/60 p-6 lg:p-8">
              <h2 className="text-xl font-semibold text-slate-900 mb-6">
                Profile Details
              </h2>

              <form onSubmit={handleSubmit} className="space-y-5">
                {/* College/School */}
                <div>
                  <label
                    htmlFor="collegeOrSchool"
                    className="block text-sm font-medium text-slate-900 mb-2"
                  >
                    College / School
                  </label>
                  <div className="relative">
                    <div className="absolute left-4 top-1/2 -translate-y-1/2">
                      <School className="h-5 w-5 text-slate-400" />
                    </div>
                    <input
                      id="collegeOrSchool"
                      name="collegeOrSchool"
                      type="text"
                      value={formData.collegeOrSchool}
                      onChange={handleChange}
                      className="block w-full pl-12 pr-4 py-3 bg-white border border-slate-200 rounded-xl transition-all focus:outline-none focus:ring-2 focus:ring-slate-900 focus:border-slate-900"
                      placeholder="Enter your institution name"
                      disabled={loading}
                    />
                  </div>
                </div>

                {/* Contact Number */}
                <div>
                  <label
                    htmlFor="contactNo"
                    className="block text-sm font-medium text-slate-900 mb-2"
                  >
                    Contact Number
                  </label>
                  <div className="relative">
                    <div className="absolute left-4 top-1/2 -translate-y-1/2">
                      <Phone className="h-5 w-5 text-slate-400" />
                    </div>
                    <input
                      id="contactNo"
                      name="contactNo"
                      type="tel"
                      value={formData.contactNo}
                      onChange={handleChange}
                      className="block w-full pl-12 pr-4 py-3 bg-white border border-slate-200 rounded-xl transition-all focus:outline-none focus:ring-2 focus:ring-slate-900 focus:border-slate-900"
                      placeholder="+1 (555) 000-0000"
                      disabled={loading}
                    />
                  </div>
                </div>

                {/* Recent Project */}
                <div>
                  <label
                    htmlFor="recentProject"
                    className="block text-sm font-medium text-slate-900 mb-2"
                  >
                    Bio
                  </label>
                  <div className="relative">
                    <div className="absolute left-4 top-4">
                      <FileText className="h-5 w-5 text-slate-400" />
                    </div>
                    <textarea
                      id="recentProject"
                      name="recentProject"
                      value={formData.recentProject}
                      onChange={handleChange}
                      rows={4}
                      className="block w-full pl-12 pr-4 py-3 bg-white border border-slate-200 rounded-xl transition-all focus:outline-none focus:ring-2 focus:ring-slate-900 focus:border-slate-900 resize-none"
                      placeholder="Describe about yourself, your interests, or your recent projects"
                      disabled={loading}
                    />
                  </div>
                </div>

                {/* Submit Button */}
                <div className="pt-4">
                  <button
                    type="submit"
                    disabled={loading}
                    className="w-full px-6 py-3 bg-slate-900 text-white rounded-xl font-medium hover:bg-slate-800 transition-all shadow-lg shadow-slate-900/25 hover:shadow-xl hover:shadow-slate-900/30 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        <span>Updating...</span>
                      </>
                    ) : (
                      <>
                        <CheckCircle2 className="w-5 h-5" />
                        <span>Save Changes</span>
                      </>
                    )}
                  </button>
                </div>
              </form>
            </div>

            {/* AI Mentor Settings Card */}
            <MentorSettings />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Profile;
