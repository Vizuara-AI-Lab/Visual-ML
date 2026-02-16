/**
 * Mentor Settings Component
 *
 * Allows users to customize AI mentor preferences from their profile page
 */

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Sparkles, Save, Loader2 } from "lucide-react";
import { useMentorStore, mentorApi } from "../features/mentor";
import type { MentorPreferences } from "../features/mentor";
import { toast } from "react-hot-toast";

export const MentorSettings: React.FC = () => {
  const { preferences, updatePreferences } = useMentorStore();
  const [localPrefs, setLocalPrefs] = useState<MentorPreferences>(preferences);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    loadPreferences();
  }, []);

  const loadPreferences = async () => {
    try {
      setLoading(true);
      const prefs = await mentorApi.getPreferences();
      setLocalPrefs(prefs);
      updatePreferences(prefs);
    } catch (error) {
      console.error("Error loading preferences:", error);
      // Use local storage prefs as fallback
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      const updated = await mentorApi.updatePreferences(localPrefs);
      updatePreferences(updated);
      toast.success("Mentor preferences saved!");
    } catch (error) {
      console.error("Error saving preferences:", error);
      toast.error("Failed to save preferences");
    } finally {
      setSaving(false);
    }
  };

  const avatarOptions = [
    { value: "scientist", label: "🧑‍🔬 Scientist", emoji: "🧑‍🔬" },
    { value: "teacher", label: "👨‍🏫 Teacher", emoji: "👨‍🏫" },
    { value: "robot", label: "🤖 Robot", emoji: "🤖" },
    { value: "wizard", label: "🧙‍♂️ Wizard", emoji: "🧙‍♂️" },
    { value: "coach", label: "👨‍💼 Coach", emoji: "👨‍💼" },
    { value: "engineer", label: "👷 Engineer", emoji: "👷" },
  ];

  if (loading) {
    return (
      <div className="bg-white rounded-2xl shadow-xl border border-slate-200/60 p-6">
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-6 h-6 text-indigo-600 animate-spin" />
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="bg-white rounded-2xl shadow-xl border border-slate-200/60 p-6"
    >
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <div
          className="w-10 h-10 rounded-xl bg-linear-to-br from-indigo-500 to-purple-600 
                        flex items-center justify-center shadow-md"
        >
          <Sparkles className="w-5 h-5 text-white" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-slate-900">
            AI Mentor Settings
          </h2>
          <p className="text-sm text-slate-600">
            Customize your learning assistant
          </p>
        </div>
      </div>

      <div className="space-y-6">
        {/* Enable/Disable */}
        <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl">
          <div>
            <label className="text-sm font-medium text-slate-900">
              Enable AI Mentor
            </label>
            <p className="text-xs text-slate-600 mt-0.5">
              Show mentor in playground
            </p>
          </div>
          <button
            onClick={() =>
              setLocalPrefs({ ...localPrefs, enabled: !localPrefs.enabled })
            }
            className={`relative w-14 h-8 rounded-full transition-colors ${
              localPrefs.enabled ? "bg-indigo-600" : "bg-slate-300"
            }`}
          >
            <motion.div
              className="absolute top-1 left-1 w-6 h-6 bg-white rounded-full shadow-md"
              animate={{ x: localPrefs.enabled ? 24 : 0 }}
              transition={{ type: "spring", stiffness: 500, damping: 30 }}
            />
          </button>
        </div>

        {/* Avatar Selection */}
        <div>
          <label className="block text-sm font-medium text-slate-900 mb-3">
            Character Avatar
          </label>
          <div className="grid grid-cols-3 gap-3">
            {avatarOptions.map((avatar) => (
              <button
                key={avatar.value}
                onClick={() =>
                  setLocalPrefs({ ...localPrefs, avatar: avatar.value })
                }
                className={`p-3 rounded-xl border-2 transition-all ${
                  localPrefs.avatar === avatar.value
                    ? "border-indigo-600 bg-indigo-50"
                    : "border-slate-200 hover:border-slate-300 bg-white"
                }`}
              >
                <div className="text-3xl mb-1">{avatar.emoji}</div>
                <div className="text-xs font-medium text-slate-700">
                  {avatar.label.split(" ")[1]}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Personality */}
        <div>
          <label className="block text-sm font-medium text-slate-900 mb-2">
            Personality Style
          </label>
          <select
            value={localPrefs.personality}
            onChange={(e) =>
              setLocalPrefs({
                ...localPrefs,
                personality: e.target.value as any,
              })
            }
            className="w-full px-4 py-2.5 bg-white border border-slate-200 rounded-xl 
                       transition-all focus:outline-none focus:ring-2 focus:ring-indigo-600 
                       focus:border-indigo-600"
          >
            <option value="encouraging">🎉 Encouraging & Supportive</option>
            <option value="professional">💼 Professional & Concise</option>
            <option value="concise">⚡ Brief & Direct</option>
            <option value="educational">📚 Detailed & Educational</option>
          </select>
        </div>

        {/* Voice Mode */}
        <div>
          <label className="block text-sm font-medium text-slate-900 mb-2">
            Voice Interaction
          </label>
          <select
            value={localPrefs.voice_mode}
            onChange={(e) =>
              setLocalPrefs({
                ...localPrefs,
                voice_mode: e.target.value as any,
              })
            }
            className="w-full px-4 py-2.5 bg-white border border-slate-200 rounded-xl 
                       transition-all focus:outline-none focus:ring-2 focus:ring-indigo-600 
                       focus:border-indigo-600"
          >
            <option value="voice_first">🔊 Voice First (Auto-play)</option>
            <option value="text_first">📝 Text First (Manual voice)</option>
            <option value="ask_each_time">❓ Ask Each Time</option>
          </select>
        </div>

        {/* Expertise Level */}
        <div>
          <label className="block text-sm font-medium text-slate-900 mb-2">
            Your ML Expertise
          </label>
          <select
            value={localPrefs.expertise_level}
            onChange={(e) =>
              setLocalPrefs({
                ...localPrefs,
                expertise_level: e.target.value as any,
              })
            }
            className="w-full px-4 py-2.5 bg-white border border-slate-200 rounded-xl 
                       transition-all focus:outline-none focus:ring-2 focus:ring-indigo-600 
                       focus:border-indigo-600"
          >
            <option value="beginner">🌱 Beginner</option>
            <option value="intermediate">🌿 Intermediate</option>
            <option value="advanced">🌳 Advanced</option>
          </select>
        </div>

        {/* Additional Options */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-slate-900">
              Show Learning Tips
            </label>
            <input
              type="checkbox"
              checked={localPrefs.show_tips}
              onChange={(e) =>
                setLocalPrefs({ ...localPrefs, show_tips: e.target.checked })
              }
              className="w-5 h-5 rounded border-slate-300 text-indigo-600 
                         focus:ring-indigo-600 focus:ring-offset-0"
            />
          </div>

          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-slate-900">
              Auto-Analyze Datasets
            </label>
            <input
              type="checkbox"
              checked={localPrefs.auto_analyze}
              onChange={(e) =>
                setLocalPrefs({ ...localPrefs, auto_analyze: e.target.checked })
              }
              className="w-5 h-5 rounded border-slate-300 text-indigo-600 
                         focus:ring-indigo-600 focus:ring-offset-0"
            />
          </div>
        </div>

        {/* Save Button */}
        <div className="pt-4">
          <button
            onClick={handleSave}
            disabled={saving}
            className="w-full px-6 py-3 bg-linear-to-r from-indigo-600 to-purple-600 
                       text-white rounded-xl font-medium hover:from-indigo-700 hover:to-purple-700 
                       transition-all shadow-lg shadow-indigo-500/30 hover:shadow-xl 
                       hover:shadow-indigo-500/40 disabled:opacity-50 disabled:cursor-not-allowed 
                       flex items-center justify-center gap-2"
          >
            {saving ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Saving...</span>
              </>
            ) : (
              <>
                <Save className="w-5 h-5" />
                <span>Save Mentor Preferences</span>
              </>
            )}
          </button>
        </div>
      </div>
    </motion.div>
  );
};
