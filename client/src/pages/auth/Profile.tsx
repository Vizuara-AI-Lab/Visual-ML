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
        className={`flex items-center gap-3 p-3.5 rounded-lg border transition-all text-left ${
          isSelected
            ? "border-amber-400 bg-amber-50 ring-1 ring-amber-200"
            : "border-slate-200 bg-white hover:border-amber-200 hover:bg-amber-50/30"
        }`}
      >
        <div
          className={`w-9 h-9 rounded-full flex items-center justify-center shrink-0 ${
            isSelected
              ? "bg-amber-500 text-white"
              : "bg-slate-100 text-slate-500"
          }`}
        >
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-slate-900">{name}</p>
          <p className="text-xs text-slate-500">{description}</p>
        </div>
        {isSelected && (
          <span className="text-[11px] font-semibold text-amber-700 bg-amber-100 px-2 py-0.5 rounded-full border border-amber-200 shrink-0">
            Active
          </span>
        )}
        <button
          onClick={(e) => {
            e.stopPropagation();
            handlePreviewVoice(voiceId);
          }}
          className={`p-1.5 rounded-md transition-colors shrink-0 ${
            previewingVoice === voiceId
              ? "text-amber-600 bg-amber-100"
              : "text-slate-400 hover:text-amber-600 hover:bg-amber-50"
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
        <div className="flex items-center gap-3">
          <Loader2 className="w-6 h-6 animate-spin text-slate-600" />
          <span className="text-slate-600">Loading profile...</span>
        </div>
      </div>
    );
  }

  const selectedVoiceId = preferences.voice_id || DEFAULT_VOICE_ID;

  return (
    <div className="min-h-screen bg-white pt-20">
      <Navbar variant="profile" />

      {/* ── Alerts (floating on top) ────────────────────────────── */}
      <div className="fixed top-24 left-1/2 -translate-x-1/2 z-50 w-full max-w-lg px-4">
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mb-3 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3 shadow-lg"
            >
              <AlertCircle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
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
              className="mb-3 p-4 bg-green-50 border border-green-200 rounded-lg flex items-start gap-3 shadow-lg"
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
      </div>

      {/* ═══════════════════════════════════════════════════════════
          SECTION 1 — Profile Hero (amber band)
          ═══════════════════════════════════════════════════════════ */}
      <div className="bg-amber-50 border-b border-amber-100">
        <div className="max-w-6xl mx-auto px-6 sm:px-8 pt-6 pb-10 lg:pt-8 lg:pb-14">
          {/* Back */}
          <button
            onClick={() => navigate("/dashboard")}
            className="flex items-center gap-2 text-amber-800/60 hover:text-amber-900 transition-colors mb-8 group"
          >
            <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
            <span className="text-sm font-medium">Back to Dashboard</span>
          </button>

          <div className="flex flex-col sm:flex-row items-start gap-6 lg:gap-8">
            {/* Avatar */}
            <div className="relative shrink-0 self-center sm:self-start">
              <div className="w-24 h-24 lg:w-28 lg:h-28 rounded-2xl overflow-hidden bg-amber-100 border-2 border-amber-200/80">
                {previewImage ? (
                  <img
                    src={previewImage}
                    alt="Profile"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center bg-slate-800">
                    <User className="w-12 h-12 text-white" />
                  </div>
                )}
              </div>

              {previewImage && !uploadingImage && (
                <button
                  onClick={handleRemoveImage}
                  className="absolute -top-2 -right-2 w-7 h-7 bg-red-500 text-white rounded-full flex items-center justify-center hover:bg-red-600 transition-colors shadow-sm"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}

              {uploadingImage && (
                <div className="absolute inset-0 bg-slate-900/50 rounded-2xl flex flex-col items-center justify-center">
                  <Loader2 className="w-6 h-6 text-white animate-spin" />
                  <span className="text-[10px] text-white font-medium mt-1">
                    {uploadProgress}%
                  </span>
                </div>
              )}
            </div>

            {/* Info */}
            <div className="flex-1 text-center sm:text-left min-w-0">
              <div className="flex flex-wrap items-center justify-center sm:justify-start gap-3 mb-1">
                <h1 className="text-2xl lg:text-3xl font-bold text-slate-900">
                  {profile.fullName || profile.emailId.split("@")[0]}
                </h1>
                {profile.isPremium && (
                  <span className="inline-flex items-center gap-1 px-2.5 py-1 bg-amber-200/60 text-amber-800 rounded-full text-xs font-semibold border border-amber-300/50">
                    <Crown className="w-3 h-3" />
                    Premium
                  </span>
                )}
              </div>

              <div className="flex items-center justify-center sm:justify-start gap-2 text-sm text-amber-800/60 mb-5">
                <Mail className="w-3.5 h-3.5" />
                <span>{profile.emailId}</span>
              </div>

              {/* Level + XP */}
              <div className="flex items-center justify-center sm:justify-start gap-3 mb-6">
                <LevelBadge size="md" />
                <div className="flex-1 max-w-xs">
                  <XPBar />
                </div>
              </div>

              {/* Actions */}
              <div className="flex flex-wrap items-center justify-center sm:justify-start gap-3">
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
                  className="inline-flex items-center gap-2 px-4 py-2 bg-slate-900 text-white rounded-lg text-sm font-medium hover:bg-slate-800 transition-colors disabled:opacity-50"
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

                <button
                  onClick={() => navigate("/auth/change-password")}
                  className="inline-flex items-center gap-2 px-4 py-2 bg-white text-slate-700 rounded-lg text-sm font-medium border border-amber-200 hover:bg-amber-50 transition-colors"
                >
                  Change Password
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════
          SECTION 2 — Profile Details (white band)
          ═══════════════════════════════════════════════════════════ */}
      <div className="bg-white border-b border-slate-100">
        <div className="max-w-6xl mx-auto px-6 sm:px-8 py-10 lg:py-14">
          <h2 className="text-xl font-bold text-slate-900 mb-6">
            Profile Details
          </h2>

          <form onSubmit={handleSubmit}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-5">
              <div>
                <label
                  htmlFor="collegeOrSchool"
                  className="block text-sm font-medium text-slate-700 mb-1.5"
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
                    className="block w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-lg text-sm transition-all focus:outline-none focus:ring-2 focus:ring-amber-400 focus:border-amber-400 focus:bg-white"
                    placeholder="Enter your institution name"
                    disabled={loading}
                  />
                </div>
              </div>

              <div>
                <label
                  htmlFor="contactNo"
                  className="block text-sm font-medium text-slate-700 mb-1.5"
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
                    className="block w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-lg text-sm transition-all focus:outline-none focus:ring-2 focus:ring-amber-400 focus:border-amber-400 focus:bg-white"
                    placeholder="+1 (555) 000-0000"
                    disabled={loading}
                  />
                </div>
              </div>
            </div>

            <div className="mb-5">
              <label
                htmlFor="recentProject"
                className="block text-sm font-medium text-slate-700 mb-1.5"
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
                  className="block w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-lg text-sm transition-all focus:outline-none focus:ring-2 focus:ring-amber-400 focus:border-amber-400 focus:bg-white resize-none"
                  placeholder="Describe about yourself, your interests, or your recent projects"
                  disabled={loading}
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="inline-flex items-center gap-2 px-6 py-2.5 bg-slate-900 text-white rounded-lg text-sm font-medium hover:bg-slate-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Updating...</span>
                </>
              ) : (
                <>
                  <CheckCircle2 className="w-4 h-4" />
                  <span>Save Changes</span>
                </>
              )}
            </button>
          </form>
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════
          SECTION 3 — Mentor Voice (slate band)
          ═══════════════════════════════════════════════════════════ */}
      <div className="bg-slate-50 border-b border-slate-100">
        <div className="max-w-6xl mx-auto px-6 sm:px-8 py-10 lg:py-14">
          <div className="flex items-center gap-3 mb-1">
            <div className="w-9 h-9 rounded-lg bg-amber-100 flex items-center justify-center">
              <Volume2 className="w-5 h-5 text-amber-600" />
            </div>
            <h2 className="text-xl font-bold text-slate-900">Mentor Voice</h2>
          </div>
          <p className="text-sm text-slate-500 mb-8 ml-12">
            Choose a voice for your AI mentor. Click play to preview, then
            select the one you like.
          </p>

          {/* Default voice */}
          <div className="mb-6">
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
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
          <div className="mb-6">
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
              Preset Voices
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
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
            <div className="mb-6">
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                Your Cloned Voices
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
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

          {/* Clone your voice */}
          <div className="mt-8 border border-dashed border-amber-300 bg-amber-50/50 rounded-xl p-6">
            <div className="flex items-center gap-2 mb-2">
              <Mic className="w-4 h-4 text-amber-600" />
              <p className="text-sm font-semibold text-slate-900">
                Clone Your Voice
              </p>
            </div>
            <p className="text-xs text-slate-500 mb-5">
              Upload a clear audio recording of your voice (WAV or MP3, max
              10MB). Speak naturally for at least 10 seconds.
            </p>

            <div className="flex flex-col sm:flex-row gap-3">
              <input
                type="text"
                value={cloneVoiceName}
                onChange={(e) => setCloneVoiceName(e.target.value)}
                placeholder="Voice name (e.g., My Voice)"
                className="flex-1 px-4 py-2.5 bg-white border border-amber-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-amber-400 focus:border-amber-400"
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
                className="inline-flex items-center justify-center gap-2 px-4 py-2.5 bg-white text-slate-700 rounded-lg text-sm font-medium border border-amber-200 hover:bg-amber-50 transition-colors disabled:opacity-50"
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
                className="inline-flex items-center justify-center gap-2 px-5 py-2.5 bg-amber-500 text-white rounded-lg text-sm font-medium hover:bg-amber-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {cloningVoice ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Cloning...</span>
                  </>
                ) : (
                  <span>Clone Voice</span>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════
          SECTION 4 — Badges (white band)
          ═══════════════════════════════════════════════════════════ */}
      <div className="bg-white">
        <div className="max-w-6xl mx-auto px-6 sm:px-8 py-10 lg:py-14">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-9 h-9 rounded-lg bg-amber-100 flex items-center justify-center">
              <Award className="w-5 h-5 text-amber-600" />
            </div>
            <h2 className="text-xl font-bold text-slate-900">Badges</h2>
          </div>
          <BadgeGrid badges={badges} />
        </div>
      </div>
    </div>
  );
};

export default Profile;
