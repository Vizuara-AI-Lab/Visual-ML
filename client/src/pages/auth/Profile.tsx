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
  Award,
  Volume2,
  Mic,
  Upload,
  Play,
  Square,
  Shield,
  Calendar,
  Sparkles,
  Settings,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import axiosInstance from "../../lib/axios";
import Navbar from "../../landingpage/Navbar";
import { useGamificationProfile } from "../../features/gamification/hooks/useGamification";
import XPBar from "../../features/gamification/components/XPBar";
import LevelBadge from "../../features/gamification/components/LevelBadge";
import BadgeGrid from "../../features/gamification/components/BadgeGrid";
import { useGamificationStore } from "../../features/gamification/store/gamificationStore";
import { useMentorStore } from "../../features/mentor/store/mentorStore";
import { mentorApi } from "../../features/mentor/api/mentorApi";

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

// ── Preset Inworld voices ─────────────────────────────────────────

const PRESET_VOICES = [
  { id: "Blake", name: "Blake", description: "Warm and friendly" },
  { id: "Clive", name: "Clive", description: "Clear and confident" },
  { id: "Hades", name: "Hades", description: "Deep and commanding" },
  { id: "Hana", name: "Hana", description: "Soft and calm" },
  { id: "Mark", name: "Mark", description: "Natural and balanced" },
  { id: "Olivia", name: "Olivia", description: "Bright and expressive" },
  { id: "Theodore", name: "Theodore", description: "Professional and steady" },
  { id: "Manoj", name: "Manoj", description: "Warm and engaging" },
] as const;

const DEFAULT_VOICE_ID =
  "default-tdxiowf-g_jzcmgci-i_iw__rajat_sir_voice_clone";

const Profile: React.FC = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const voiceFileRef = useRef<HTMLInputElement>(null);
  const previewAudioRef = useRef<HTMLAudioElement | null>(null);

  useGamificationProfile();
  const badges = useGamificationStore((s) => s.badges);

  const { preferences, updatePreferences } = useMentorStore();

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

  // Password state
  const [passwordData, setPasswordData] = useState({ oldPassword: "", newPassword: "", confirmPassword: "" });
  const [changingPassword, setChangingPassword] = useState(false);

  // Voice state
  const [previewingVoice, setPreviewingVoice] = useState<string | null>(null);
  const [cloneVoiceName, setCloneVoiceName] = useState("");
  const [cloningVoice, setCloningVoice] = useState(false);
  const [clonedVoices, setClonedVoices] = useState<
    Array<{ id: string; name: string }>
  >(() => {
    try {
      const saved = localStorage.getItem("cloned-voices");
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  // Active section for sidebar nav
  const [activeSection, setActiveSection] = useState<string>("details");

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

  const handleImageUpload = async (
    e: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("Please select an image file");
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      setError("Image size must be less than 5MB");
      return;
    }

    setUploadingImage(true);
    setError("");
    setUploadProgress(0);

    try {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewImage(reader.result as string);
      };
      reader.readAsDataURL(file);

      const formData = new FormData();
      formData.append("file", file);

      const response = await axiosInstance.post(
        "/auth/student/upload-profile-pic",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
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
      const response = await axiosInstance.patch(
        "/auth/student/me",
        formData,
      );
      setProfile(response.data);

      const existingUser = localStorage.getItem("user");
      if (existingUser) {
        const userData = JSON.parse(existingUser);
        userData.profilePic =
          response.data.profilePic || formData.profilePic;
        userData.collegeOrSchool =
          response.data.collegeOrSchool || formData.collegeOrSchool;
        userData.fullName = response.data.fullName || userData.fullName;
        localStorage.setItem("user", JSON.stringify(userData));
      }

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

  const handlePasswordChange = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    if (passwordData.newPassword.length < 6) {
      setError("New password must be at least 6 characters");
      return;
    }
    if (passwordData.newPassword !== passwordData.confirmPassword) {
      setError("New passwords do not match");
      return;
    }

    setChangingPassword(true);
    try {
      await axiosInstance.post("/auth/student/change-password", {
        oldPassword: passwordData.oldPassword,
        newPassword: passwordData.newPassword,
      });
      setPasswordData({ oldPassword: "", newPassword: "", confirmPassword: "" });
      setSuccess("Password changed successfully! Please log in again.");
      // Backend clears cookies, so redirect to login after a moment
      setTimeout(() => navigate("/login"), 2000);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to change password");
    } finally {
      setChangingPassword(false);
    }
  };

  // ── Voice handlers ────────────────────────────────────────────

  const handleVoiceSelect = (voiceId: string) => {
    updatePreferences({ voice_id: voiceId });
  };

  const handlePreviewVoice = async (voiceId: string) => {
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
      previewAudioRef.current = null;
    }

    if (previewingVoice === voiceId) {
      setPreviewingVoice(null);
      return;
    }

    setPreviewingVoice(voiceId);
    try {
      const response = await mentorApi.previewVoice(voiceId);
      if (response.success && response.audio_base64) {
        const binaryString = atob(response.audio_base64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: "audio/ogg; codecs=opus" });
        const audioUrl = URL.createObjectURL(blob);

        previewAudioRef.current = new Audio(audioUrl);
        previewAudioRef.current.onended = () => {
          setPreviewingVoice(null);
          URL.revokeObjectURL(audioUrl);
        };
        previewAudioRef.current.onerror = () => {
          setPreviewingVoice(null);
          URL.revokeObjectURL(audioUrl);
        };
        await previewAudioRef.current.play();
      } else {
        setPreviewingVoice(null);
      }
    } catch {
      setPreviewingVoice(null);
    }
  };

  const handleCloneVoice = async () => {
    const file = voiceFileRef.current?.files?.[0];
    if (!file || !cloneVoiceName.trim()) return;

    setCloningVoice(true);
    setError("");
    try {
      const result = await mentorApi.cloneVoice(
        cloneVoiceName.trim(),
        file,
      );
      if (result.success) {
        const newVoice = {
          id: result.voice_id,
          name: result.display_name,
        };
        const updated = [...clonedVoices, newVoice];
        setClonedVoices(updated);
        localStorage.setItem("cloned-voices", JSON.stringify(updated));
        updatePreferences({ voice_id: result.voice_id });
        setCloneVoiceName("");
        if (voiceFileRef.current) voiceFileRef.current.value = "";
        setSuccess("Voice cloned successfully!");
        setTimeout(() => setSuccess(""), 3000);
      }
    } catch (err: any) {
      setError(
        err.response?.data?.detail || "Voice cloning failed. Try again.",
      );
    } finally {
      setCloningVoice(false);
    }
  };

  // ── Voice item renderer ───────────────────────────────────────

  const renderVoiceItem = (
    voiceId: string,
    name: string,
    description: string,
    icon: React.ReactNode,
  ) => {
    const isSelected = selectedVoiceId === voiceId;
    return (
      <button
        key={voiceId}
        onClick={() => handleVoiceSelect(voiceId)}
        className={`flex items-center gap-3 p-3 rounded-xl border-2 transition-all text-left w-full ${
          isSelected
            ? "border-emerald-400 bg-emerald-50 shadow-sm"
            : "border-transparent bg-white hover:border-slate-200 hover:shadow-sm"
        }`}
      >
        <div
          className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${
            isSelected
              ? "bg-emerald-500 text-white"
              : "bg-slate-100 text-slate-500"
          }`}
        >
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-slate-800">{name}</p>
          <p className="text-xs text-slate-500">{description}</p>
        </div>
        {isSelected && (
          <span className="text-[10px] font-bold text-emerald-700 bg-emerald-100 px-2 py-0.5 rounded-full shrink-0">
            Active
          </span>
        )}
        <button
          onClick={(e) => {
            e.stopPropagation();
            handlePreviewVoice(voiceId);
          }}
          className={`p-2 rounded-lg transition-colors shrink-0 ${
            previewingVoice === voiceId
              ? "text-emerald-600 bg-emerald-100"
              : "text-slate-400 hover:text-emerald-600 hover:bg-emerald-50"
          }`}
          title="Preview voice"
        >
          {previewingVoice === voiceId ? (
            <Square className="w-4 h-4" />
          ) : (
            <Play className="w-4 h-4" />
          )}
        </button>
      </button>
    );
  };

  if (!profile) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="flex flex-col items-center gap-4"
        >
          <div className="w-12 h-12 rounded-2xl bg-emerald-100 flex items-center justify-center">
            <Loader2 className="w-6 h-6 animate-spin text-emerald-600" />
          </div>
          <p className="text-sm font-medium text-slate-500">Loading profile...</p>
        </motion.div>
      </div>
    );
  }

  const selectedVoiceId = preferences.voice_id || DEFAULT_VOICE_ID;

  const sidebarSections = [
    { id: "details", label: "Profile", icon: User },
    { id: "security", label: "Security", icon: Shield },
    { id: "voice", label: "Voice", icon: Volume2 },
    { id: "badges", label: "Badges", icon: Award },
  ];

  return (
    <div className="min-h-screen bg-slate-50 pt-20">
      <Navbar variant="profile" />

      {/* ── Alerts ──────────────────────────────────── */}
      <div className="fixed top-24 left-1/2 -translate-x-1/2 z-50 w-full max-w-lg px-4">
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              className="mb-3 p-4 bg-red-50 border border-red-200 rounded-xl flex items-start gap-3 shadow-lg"
            >
              <AlertCircle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="text-sm font-semibold text-red-900">Error</p>
                <p className="text-sm text-red-700">{error}</p>
              </div>
              <button
                onClick={() => setError("")}
                className="text-red-400 hover:text-red-600 p-0.5"
              >
                <X className="w-4 h-4" />
              </button>
            </motion.div>
          )}

          {success && (
            <motion.div
              initial={{ opacity: 0, y: -10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              className="mb-3 p-4 bg-emerald-50 border border-emerald-200 rounded-xl flex items-start gap-3 shadow-lg"
            >
              <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="text-sm font-semibold text-emerald-900">Success</p>
                <p className="text-sm text-emerald-700">{success}</p>
              </div>
              <button
                onClick={() => setSuccess("")}
                className="text-emerald-400 hover:text-emerald-600 p-0.5"
              >
                <X className="w-4 h-4" />
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Back button */}
        <button
          onClick={() => navigate("/dashboard")}
          className="flex items-center gap-2 text-slate-500 hover:text-slate-800 transition-colors mb-6 group"
        >
          <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
          <span className="text-sm font-medium">Back to Dashboard</span>
        </button>

        {/* ═══════════════════════════════════════════════════════════
            HERO CARD
            ═══════════════════════════════════════════════════════════ */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden mb-6"
        >
          {/* Gradient banner */}
          <div className="h-32 bg-linear-to-r from-emerald-500 to-emerald-600 relative">
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiNmZmZmZmYiIGZpbGwtb3BhY2l0eT0iMC4wNSI+PHBhdGggZD0iTTM2IDE4YzMuMzEzIDAgNiAyLjY4NyA2IDZzLTIuNjg3IDYtNiA2LTYtMi42ODctNi02IDIuNjg3LTYgNi02ek0xOCAzNmMzLjMxMyAwIDYgMi42ODcgNiA2cy0yLjY4NyA2LTYgNi02LTIuNjg3LTYtNiAyLjY4Ny02IDYtNnoiLz48L2c+PC9nPjwvc3ZnPg==')] opacity-30" />
            {profile.isPremium && (
              <div className="absolute top-4 right-4 flex items-center gap-1.5 px-3 py-1.5 bg-white/20 backdrop-blur-sm rounded-full text-white text-xs font-semibold">
                <Crown className="w-3.5 h-3.5" />
                Premium
              </div>
            )}
          </div>

          <div className="px-6 sm:px-8 pb-6 -mt-14 relative">
            <div className="flex flex-col sm:flex-row items-center sm:items-end gap-5">
              {/* Avatar */}
              <div className="relative shrink-0">
                <div className="w-28 h-28 rounded-2xl overflow-hidden border-4 border-white shadow-lg bg-slate-800">
                  {previewImage ? (
                    <img
                      src={previewImage}
                      alt="Profile"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <User className="w-12 h-12 text-white/60" />
                    </div>
                  )}
                </div>

                {previewImage && !uploadingImage && (
                  <button
                    onClick={handleRemoveImage}
                    className="absolute -top-1.5 -right-1.5 w-6 h-6 bg-red-500 text-white rounded-full flex items-center justify-center hover:bg-red-600 transition-colors shadow-md"
                  >
                    <X className="w-3 h-3" />
                  </button>
                )}

                {uploadingImage && (
                  <div className="absolute inset-0 bg-slate-900/60 rounded-2xl flex flex-col items-center justify-center border-4 border-white">
                    <Loader2 className="w-6 h-6 text-white animate-spin" />
                    <span className="text-[10px] text-white font-bold mt-1">
                      {uploadProgress}%
                    </span>
                  </div>
                )}

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
                  className="absolute -bottom-1 -right-1 w-8 h-8 bg-emerald-500 text-white rounded-xl flex items-center justify-center hover:bg-emerald-600 transition-colors shadow-md border-2 border-white"
                >
                  <Camera className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Name + meta */}
              <div className="flex-1 text-center sm:text-left sm:pb-1 min-w-0">
                <h1 className="text-2xl font-bold text-slate-900">
                  {profile.fullName || profile.emailId.split("@")[0]}
                </h1>
                <div className="flex flex-wrap items-center justify-center sm:justify-start gap-x-4 gap-y-1 mt-1.5">
                  <span className="flex items-center gap-1.5 text-sm text-slate-500">
                    <Mail className="w-3.5 h-3.5" />
                    {profile.emailId}
                  </span>
                  {formData.collegeOrSchool && (
                    <span className="flex items-center gap-1.5 text-sm text-slate-500">
                      <School className="w-3.5 h-3.5" />
                      {formData.collegeOrSchool}
                    </span>
                  )}
                </div>
              </div>

              {/* Level + XP */}
              <div className="flex items-center gap-4 sm:pb-1">
                <LevelBadge size="lg" />
                <div className="w-40">
                  <XPBar />
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* ═══════════════════════════════════════════════════════════
            CONTENT — sidebar + main area
            ═══════════════════════════════════════════════════════════ */}
        <div className="flex flex-col lg:flex-row gap-6">
          {/* Sidebar nav */}
          <motion.div
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
            className="lg:w-56 shrink-0"
          >
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-2 lg:sticky lg:top-28">
              <nav className="flex lg:flex-col gap-1">
                {sidebarSections.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => setActiveSection(s.id)}
                    className={`flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all w-full text-left ${
                      activeSection === s.id
                        ? "bg-emerald-50 text-emerald-700 shadow-sm"
                        : "text-slate-600 hover:bg-slate-50 hover:text-slate-800"
                    }`}
                  >
                    <s.icon className={`w-4 h-4 ${activeSection === s.id ? "text-emerald-500" : "text-slate-400"}`} />
                    {s.label}
                  </button>
                ))}

              </nav>
            </div>
          </motion.div>

          {/* Main content */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.15 }}
            className="flex-1 min-w-0"
          >
            <AnimatePresence mode="wait">
              {/* ── DETAILS SECTION ──────────────────── */}
              {activeSection === "details" && (
                <motion.div
                  key="details"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.2 }}
                  className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden"
                >
                  <div className="px-6 py-5 border-b border-slate-100 flex items-center gap-3">
                    <div className="w-9 h-9 rounded-xl bg-emerald-50 flex items-center justify-center">
                      <Settings className="w-4 h-4 text-emerald-600" />
                    </div>
                    <div>
                      <h2 className="text-base font-bold text-slate-900">Profile Details</h2>
                      <p className="text-xs text-slate-500">Update your personal information</p>
                    </div>
                  </div>

                  <form onSubmit={handleSubmit} className="p-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-5">
                      <div>
                        <label
                          htmlFor="collegeOrSchool"
                          className="block text-sm font-medium text-slate-700 mb-2"
                        >
                          College / School
                        </label>
                        <div className="relative">
                          <div className="absolute left-3.5 top-1/2 -translate-y-1/2">
                            <School className="h-4 w-4 text-slate-400" />
                          </div>
                          <input
                            id="collegeOrSchool"
                            name="collegeOrSchool"
                            type="text"
                            value={formData.collegeOrSchool}
                            onChange={handleChange}
                            className="block w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-sm transition-all focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:border-emerald-400 focus:bg-white"
                            placeholder="Enter your institution name"
                            disabled={loading}
                          />
                        </div>
                      </div>

                      <div>
                        <label
                          htmlFor="contactNo"
                          className="block text-sm font-medium text-slate-700 mb-2"
                        >
                          Contact Number
                        </label>
                        <div className="relative">
                          <div className="absolute left-3.5 top-1/2 -translate-y-1/2">
                            <Phone className="h-4 w-4 text-slate-400" />
                          </div>
                          <input
                            id="contactNo"
                            name="contactNo"
                            type="tel"
                            value={formData.contactNo}
                            onChange={handleChange}
                            className="block w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-sm transition-all focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:border-emerald-400 focus:bg-white"
                            placeholder="+1 (555) 000-0000"
                            disabled={loading}
                          />
                        </div>
                      </div>
                    </div>

                    <div className="mb-6">
                      <label
                        htmlFor="recentProject"
                        className="block text-sm font-medium text-slate-700 mb-2"
                      >
                        Bio
                      </label>
                      <div className="relative">
                        <div className="absolute left-3.5 top-3">
                          <FileText className="h-4 w-4 text-slate-400" />
                        </div>
                        <textarea
                          id="recentProject"
                          name="recentProject"
                          value={formData.recentProject}
                          onChange={handleChange}
                          rows={3}
                          className="block w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-sm transition-all focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:border-emerald-400 focus:bg-white resize-none"
                          placeholder="Describe about yourself, your interests, or your recent projects"
                          disabled={loading}
                        />
                      </div>
                    </div>

                    <button
                      type="submit"
                      disabled={loading}
                      className="inline-flex items-center gap-2 px-6 py-2.5 bg-linear-to-r from-emerald-500 to-emerald-600 text-white rounded-xl text-sm font-semibold hover:from-emerald-600 hover:to-emerald-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          Updating...
                        </>
                      ) : (
                        <>
                          <CheckCircle2 className="w-4 h-4" />
                          Save Changes
                        </>
                      )}
                    </button>
                  </form>
                </motion.div>
              )}

              {/* ── SECURITY SECTION ──────────────────── */}
              {activeSection === "security" && (
                <motion.div
                  key="security"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.2 }}
                  className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden"
                >
                  <div className="px-6 py-5 border-b border-slate-100 flex items-center gap-3">
                    <div className="w-9 h-9 rounded-xl bg-emerald-50 flex items-center justify-center">
                      <Shield className="w-4 h-4 text-emerald-600" />
                    </div>
                    <div>
                      <h2 className="text-base font-bold text-slate-900">Change Password</h2>
                      <p className="text-xs text-slate-500">Update your account password</p>
                    </div>
                  </div>

                  <form onSubmit={handlePasswordChange} className="p-6 space-y-5">
                    <div>
                      <label className="block text-sm font-medium text-slate-700 mb-2">
                        Current Password
                      </label>
                      <input
                        type="password"
                        value={passwordData.oldPassword}
                        onChange={(e) => setPasswordData((p) => ({ ...p, oldPassword: e.target.value }))}
                        className="block w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-sm transition-all focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:border-emerald-400 focus:bg-white"
                        placeholder="Enter current password"
                        required
                        disabled={changingPassword}
                      />
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                      <div>
                        <label className="block text-sm font-medium text-slate-700 mb-2">
                          New Password
                        </label>
                        <input
                          type="password"
                          value={passwordData.newPassword}
                          onChange={(e) => setPasswordData((p) => ({ ...p, newPassword: e.target.value }))}
                          className="block w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-sm transition-all focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:border-emerald-400 focus:bg-white"
                          placeholder="At least 6 characters"
                          required
                          minLength={6}
                          disabled={changingPassword}
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-slate-700 mb-2">
                          Confirm New Password
                        </label>
                        <input
                          type="password"
                          value={passwordData.confirmPassword}
                          onChange={(e) => setPasswordData((p) => ({ ...p, confirmPassword: e.target.value }))}
                          className={`block w-full px-4 py-2.5 bg-slate-50 border rounded-xl text-sm transition-all focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:border-emerald-400 focus:bg-white ${
                            passwordData.confirmPassword && passwordData.confirmPassword !== passwordData.newPassword
                              ? "border-red-300"
                              : "border-slate-200"
                          }`}
                          placeholder="Re-enter new password"
                          required
                          minLength={6}
                          disabled={changingPassword}
                        />
                        {passwordData.confirmPassword && passwordData.confirmPassword !== passwordData.newPassword && (
                          <p className="text-xs text-red-500 mt-1.5">Passwords do not match</p>
                        )}
                      </div>
                    </div>

                    <button
                      type="submit"
                      disabled={changingPassword || !passwordData.oldPassword || !passwordData.newPassword || !passwordData.confirmPassword}
                      className="inline-flex items-center gap-2 px-6 py-2.5 bg-linear-to-r from-emerald-500 to-emerald-600 text-white rounded-xl text-sm font-semibold hover:from-emerald-600 hover:to-emerald-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
                    >
                      {changingPassword ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          Changing...
                        </>
                      ) : (
                        <>
                          <Shield className="w-4 h-4" />
                          Change Password
                        </>
                      )}
                    </button>
                  </form>
                </motion.div>
              )}

              {/* ── VOICE SECTION ──────────────────── */}
              {activeSection === "voice" && (
                <motion.div
                  key="voice"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.2 }}
                  className="space-y-6"
                >
                  <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
                    <div className="px-6 py-5 border-b border-slate-100 flex items-center gap-3">
                      <div className="w-9 h-9 rounded-xl bg-emerald-50 flex items-center justify-center">
                        <Volume2 className="w-4 h-4 text-emerald-600" />
                      </div>
                      <div>
                        <h2 className="text-base font-bold text-slate-900">Mentor Voice</h2>
                        <p className="text-xs text-slate-500">
                          Choose a voice for your AI mentor
                        </p>
                      </div>
                    </div>

                    <div className="p-6 space-y-6">
                      {/* Default voice */}
                      <div>
                        <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
                          Default
                        </p>
                        <div className="max-w-xl">
                          {renderVoiceItem(
                            DEFAULT_VOICE_ID,
                            "Rajat Sir",
                            "Original cloned voice",
                            <Volume2 className="w-4 h-4" />,
                          )}
                        </div>
                      </div>

                      {/* Preset voices */}
                      <div>
                        <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
                          Preset Voices
                        </p>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                          {PRESET_VOICES.map((voice) =>
                            renderVoiceItem(
                              voice.id,
                              voice.name,
                              voice.description,
                              <Volume2 className="w-4 h-4" />,
                            ),
                          )}
                        </div>
                      </div>

                      {/* Cloned voices */}
                      {clonedVoices.length > 0 && (
                        <div>
                          <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
                            Your Cloned Voices
                          </p>
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                            {clonedVoices.map((voice) =>
                              renderVoiceItem(
                                voice.id,
                                voice.name,
                                "Custom clone",
                                <Mic className="w-4 h-4" />,
                              ),
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Clone your voice card */}
                  <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
                    <div className="px-6 py-5 border-b border-slate-100 flex items-center gap-3">
                      <div className="w-9 h-9 rounded-xl bg-amber-50 flex items-center justify-center">
                        <Mic className="w-4 h-4 text-amber-600" />
                      </div>
                      <div>
                        <h2 className="text-base font-bold text-slate-900">Clone Your Voice</h2>
                        <p className="text-xs text-slate-500">
                          Upload a clear recording (WAV or MP3, max 10MB, at least 10 seconds)
                        </p>
                      </div>
                    </div>

                    <div className="p-6">
                      <div className="flex flex-col sm:flex-row gap-3">
                        <input
                          type="text"
                          value={cloneVoiceName}
                          onChange={(e) => setCloneVoiceName(e.target.value)}
                          placeholder="Voice name (e.g., My Voice)"
                          className="flex-1 px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:border-emerald-400 focus:bg-white"
                          disabled={cloningVoice}
                        />

                        <input
                          ref={voiceFileRef}
                          type="file"
                          accept="audio/*"
                          className="hidden"
                          disabled={cloningVoice}
                        />
                        <button
                          onClick={() => voiceFileRef.current?.click()}
                          disabled={cloningVoice}
                          className="inline-flex items-center justify-center gap-2 px-4 py-2.5 bg-white text-slate-700 rounded-xl text-sm font-medium border border-slate-200 hover:bg-slate-50 transition-colors disabled:opacity-50"
                        >
                          <Upload className="w-4 h-4" />
                          <span>
                            {voiceFileRef.current?.files?.[0]?.name ||
                              "Choose Audio File"}
                          </span>
                        </button>

                        <button
                          onClick={handleCloneVoice}
                          disabled={cloningVoice || !cloneVoiceName.trim()}
                          className="inline-flex items-center justify-center gap-2 px-5 py-2.5 bg-linear-to-r from-emerald-500 to-emerald-600 text-white rounded-xl text-sm font-semibold hover:from-emerald-600 hover:to-emerald-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
                        >
                          {cloningVoice ? (
                            <>
                              <Loader2 className="w-4 h-4 animate-spin" />
                              Cloning...
                            </>
                          ) : (
                            <>
                              <Sparkles className="w-4 h-4" />
                              Clone Voice
                            </>
                          )}
                        </button>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* ── BADGES SECTION ──────────────────── */}
              {activeSection === "badges" && (
                <motion.div
                  key="badges"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.2 }}
                  className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden"
                >
                  <div className="px-6 py-5 border-b border-slate-100 flex items-center gap-3">
                    <div className="w-9 h-9 rounded-xl bg-amber-50 flex items-center justify-center">
                      <Award className="w-4 h-4 text-amber-600" />
                    </div>
                    <div>
                      <h2 className="text-base font-bold text-slate-900">Badges</h2>
                      <p className="text-xs text-slate-500">
                        Your achievements and milestones
                      </p>
                    </div>
                  </div>

                  <div className="p-6">
                    <BadgeGrid badges={badges} />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Profile;
