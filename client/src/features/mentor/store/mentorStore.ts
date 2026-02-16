/**
 * Mentor Store - Zustand state management for AI Mentor
 *
 * Manages mentor visibility, suggestions, audio playback, and user preferences
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface MentorAction {
  label: string;
  type:
    | "add_node"
    | "fix_issue"
    | "learn_more"
    | "execute"
    | "show_guide"
    | "dataset_guidance";
  payload?: Record<string, unknown>;
}

export interface MentorSuggestion {
  id: string;
  type:
    | "greeting"
    | "dataset_analysis"
    | "next_step"
    | "error_explanation"
    | "best_practice"
    | "learning_tip"
    | "warning";
  priority: "info" | "warning" | "critical";
  title: string;
  message: string;
  voice_text?: string;
  actions: MentorAction[];
  timestamp?: string;
  dismissible: boolean;
}

export interface MentorPreferences {
  enabled: boolean;
  avatar: string;
  personality: "encouraging" | "professional" | "concise" | "educational";
  voice_mode: "voice_first" | "text_first" | "ask_each_time";
  expertise_level: "beginner" | "intermediate" | "advanced";
  show_tips: boolean;
  auto_analyze: boolean;
}

export interface DatasetInsight {
  summary: string;
  n_rows: number;
  n_columns: number;
  numeric_columns: string[];
  categorical_columns: string[];
  missing_values: Record<string, number>;
  high_cardinality_columns: Array<{
    column: string;
    unique_values: number;
    warning: string;
  }>;
  scaling_needed: string[];
  recommended_target?: string;
  warnings: string[];
  recommendations: string[];
}

interface AudioQueueItem {
  id: string;
  text: string;
  audioData?: string; // base64
}

interface MentorState {
  // UI State
  isOpen: boolean;
  isMinimized: boolean;
  // Current guide state
  currentGuide: {
    modelType: string;
    steps: Array<{
      step: number;
      node_type: string;
      description: string;
    }>;
    explanation: string;
    estimatedTime?: string;
    currentStep: number;
  } | null;
  // Suggestions
  currentSuggestion: MentorSuggestion | null;
  suggestions: MentorSuggestion[];
  history: MentorSuggestion[];

  // Dataset insights
  datasetInsights: DatasetInsight | null;

  // Audio
  isSpeaking: boolean;
  audioQueue: AudioQueueItem[];
  currentAudioId: string | null;

  // Preferences
  preferences: MentorPreferences;

  // Actions
  setOpen: (open: boolean) => void;
  setMinimized: (minimized: boolean) => void;
  toggleOpen: () => void;
  toggleMinimized: () => void;

  // Guide actions
  showGuide: (guide: {
    modelType: string;
    steps: Array<{
      step: number;
      node_type: string;
      description: string;
    }>;
    explanation: string;
    estimatedTime?: string;
  }) => void;
  clearGuide: () => void;
  advanceGuideStep: () => void;
  setGuideStep: (step: number) => void;

  showSuggestion: (suggestion: MentorSuggestion) => void;
  dismissSuggestion: (suggestionId: string) => void;
  clearSuggestions: () => void;

  setDatasetInsights: (insights: DatasetInsight | null) => void;

  // Audio controls
  playAudio: (id: string, text: string, audioData?: string) => void;
  stopAudio: () => void;
  setIsSpeaking: (speaking: boolean) => void;
  clearAudioQueue: () => void;

  // Preferences
  updatePreferences: (prefs: Partial<MentorPreferences>) => void;
  resetPreferences: () => void;
}

const defaultPreferences: MentorPreferences = {
  enabled: true,
  avatar: "scientist",
  personality: "encouraging",
  voice_mode: "text_first",
  expertise_level: "beginner",
  show_tips: true,
  auto_analyze: true,
};

export const useMentorStore = create<MentorState>()(
  persist(
    (set, get) => ({
      // Initial state
      isOpen: true,
      isMinimized: false,
      currentGuide: null,
      currentSuggestion: null,
      suggestions: [],
      history: [],
      datasetInsights: null,
      isSpeaking: false,
      audioQueue: [],
      currentAudioId: null,
      preferences: defaultPreferences,

      // UI actions
      setOpen: (open) => set({ isOpen: open }),
      setMinimized: (minimized) => set({ isMinimized: minimized }),

      // Guide actions
      showGuide: (guide) =>
        set({
          currentGuide: { ...guide, currentStep: 0 },
          currentSuggestion: null, // Clear current suggestion when showing guide
        }),
      clearGuide: () => set({ currentGuide: null }),
      advanceGuideStep: () =>
        set((state) => ({
          currentGuide: state.currentGuide
            ? {
                ...state.currentGuide,
                currentStep: Math.min(
                  state.currentGuide.currentStep + 1,
                  state.currentGuide.steps.length - 1,
                ),
              }
            : null,
        })),
      setGuideStep: (step) =>
        set((state) => ({
          currentGuide: state.currentGuide
            ? { ...state.currentGuide, currentStep: step }
            : null,
        })),
      toggleOpen: () => set((state) => ({ isOpen: !state.isOpen })),
      toggleMinimized: () =>
        set((state) => ({ isMinimized: !state.isMinimized })),

      // Suggestion actions
      showSuggestion: (suggestion) =>
        set((state) => ({
          currentSuggestion: suggestion,
          suggestions: [
            suggestion,
            ...state.suggestions.filter((s) => s.id !== suggestion.id),
          ],
          history: [suggestion, ...state.history].slice(0, 20), // Keep last 20
          isOpen: true,
          isMinimized: false,
        })),

      dismissSuggestion: (suggestionId) =>
        set((state) => ({
          suggestions: state.suggestions.filter((s) => s.id !== suggestionId),
          currentSuggestion:
            state.currentSuggestion?.id === suggestionId
              ? state.suggestions[0] || null
              : state.currentSuggestion,
        })),

      clearSuggestions: () =>
        set({
          suggestions: [],
          currentSuggestion: null,
        }),

      setDatasetInsights: (insights) => set({ datasetInsights: insights }),

      // Audio actions
      playAudio: (id, text, audioData) =>
        set((state) => ({
          audioQueue: [...state.audioQueue, { id, text, audioData }],
          currentAudioId: id,
        })),

      stopAudio: () =>
        set({
          isSpeaking: false,
          currentAudioId: null,
        }),

      setIsSpeaking: (speaking) => set({ isSpeaking: speaking }),

      clearAudioQueue: () =>
        set({
          audioQueue: [],
          currentAudioId: null,
          isSpeaking: false,
        }),

      // Preference actions
      updatePreferences: (prefs) =>
        set((state) => ({
          preferences: { ...state.preferences, ...prefs },
        })),

      resetPreferences: () => set({ preferences: defaultPreferences }),
    }),
    {
      name: "mentor-storage",
      partialize: (state) => ({
        preferences: state.preferences,
        history: state.history.slice(0, 10), // Persist only last 10 history items
      }),
    },
  ),
);
