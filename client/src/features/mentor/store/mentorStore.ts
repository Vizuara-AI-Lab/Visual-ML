/**
 * Mentor Store - Zustand state management for AI Mentor
 *
 * Manages mentor visibility, suggestions, audio playback, user preferences,
 * and the structured learning flow state machine.
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";

// ── Learning Flow Types ──────────────────────────────────────────

export type LearningStage =
  | "welcome"
  | "algorithm_selected"
  | "prompt_drag_dataset"
  | "dataset_node_added"
  | "dataset_configured"
  | "prompt_column_info"
  | "column_info_added"
  | "prompt_run_column_info"
  | "column_info_executed"
  | "prompt_missing_values"
  | "missing_values_added"
  | "missing_values_configured"
  | "prompt_encoding"
  | "encoding_added"
  | "prompt_split"
  | "split_added"
  | "split_configured"
  | "prompt_model"
  | "model_added"
  | "prompt_metrics"
  | "metrics_added"
  | "prompt_final_run"
  | "pipeline_executed"
  | "completed"
  | "error_occurred";

export interface LearningFlowState {
  stage: LearningStage;
  selectedAlgorithm: string | null;
  previousStage: LearningStage | null;
  datasetInfo: {
    filename: string;
    nRows: number;
    nCols: number;
  } | null;
  columnInfoResults: {
    missingColumns: Array<{ column: string; count: number }>;
    categoricalColumns: string[];
    numericColumns: string[];
    totalMissing: number;
  } | null;
  trackedNodeIds: {
    dataset: string | null;
    columnInfo: string | null;
    missingHandler: string | null;
    encoding: string | null;
    split: string | null;
    model: string | null;
    metrics: string[];
  };
  executionResults: Record<string, number> | null;
}

const defaultLearningFlow: LearningFlowState = {
  stage: "welcome",
  selectedAlgorithm: null,
  previousStage: null,
  datasetInfo: null,
  columnInfoResults: null,
  trackedNodeIds: {
    dataset: null,
    columnInfo: null,
    missingHandler: null,
    encoding: null,
    split: null,
    model: null,
    metrics: [],
  },
  executionResults: null,
};

// ── Existing Types ───────────────────────────────────────────────

export interface MentorAction {
  label: string;
  type:
    | "add_node"
    | "fix_issue"
    | "learn_more"
    | "execute"
    | "show_guide"
    | "dataset_guidance"
    | "select_algorithm";
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
  voice_id: string;
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
  audioData?: string;
}

// ── Store Interface ──────────────────────────────────────────────

interface MentorState {
  // UI State
  isOpen: boolean;
  isMinimized: boolean;

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

  // Learning Flow
  learningFlow: LearningFlowState;

  // UI actions
  setOpen: (open: boolean) => void;
  setMinimized: (minimized: boolean) => void;
  toggleOpen: () => void;
  toggleMinimized: () => void;

  // Suggestion actions
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

  // Learning Flow actions
  setStage: (stage: LearningStage) => void;
  setSelectedAlgorithm: (algorithm: string) => void;
  setDatasetInfo: (info: LearningFlowState["datasetInfo"]) => void;
  setColumnInfoResults: (
    data: LearningFlowState["columnInfoResults"],
  ) => void;
  setExecutionResults: (results: Record<string, number> | null) => void;
  trackNodeId: (
    role: keyof LearningFlowState["trackedNodeIds"],
    id: string,
  ) => void;
  resetFlow: () => void;
}

// ── Defaults ─────────────────────────────────────────────────────

const defaultPreferences: MentorPreferences = {
  enabled: true,
  avatar: "scientist",
  personality: "encouraging",
  voice_mode: "text_first",
  voice_id: "default-tdxiowf-g_jzcmgci-i_iw__rajat_sir_voice_clone",
  expertise_level: "beginner",
  show_tips: true,
  auto_analyze: true,
};

// ── Store ────────────────────────────────────────────────────────

export const useMentorStore = create<MentorState>()(
  persist(
    (set) => ({
      // Initial state
      isOpen: true,
      isMinimized: false,
      currentSuggestion: null,
      suggestions: [],
      history: [],
      datasetInsights: null,
      isSpeaking: false,
      audioQueue: [],
      currentAudioId: null,
      preferences: defaultPreferences,
      learningFlow: defaultLearningFlow,

      // UI actions
      setOpen: (open) => set({ isOpen: open }),
      setMinimized: (minimized) => set({ isMinimized: minimized }),
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
          history: [suggestion, ...state.history].slice(0, 20),
          isOpen: true,
          isMinimized: false,
        })),

      dismissSuggestion: (suggestionId) =>
        set((state) => ({
          suggestions: state.suggestions.filter(
            (s) => s.id !== suggestionId,
          ),
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

      // Learning Flow actions
      setStage: (stage) =>
        set((state) => ({
          learningFlow: {
            ...state.learningFlow,
            previousStage: state.learningFlow.stage,
            stage,
          },
        })),

      setSelectedAlgorithm: (algorithm) =>
        set((state) => ({
          learningFlow: {
            ...state.learningFlow,
            selectedAlgorithm: algorithm,
          },
        })),

      setDatasetInfo: (info) =>
        set((state) => ({
          learningFlow: {
            ...state.learningFlow,
            datasetInfo: info,
          },
        })),

      setColumnInfoResults: (data) =>
        set((state) => ({
          learningFlow: {
            ...state.learningFlow,
            columnInfoResults: data,
          },
        })),

      setExecutionResults: (results) =>
        set((state) => ({
          learningFlow: {
            ...state.learningFlow,
            executionResults: results,
          },
        })),

      trackNodeId: (role, id) =>
        set((state) => {
          const tracked = { ...state.learningFlow.trackedNodeIds };
          if (role === "metrics") {
            tracked.metrics = [...tracked.metrics, id];
          } else {
            (tracked as Record<string, string | string[] | null>)[role] = id;
          }
          return {
            learningFlow: {
              ...state.learningFlow,
              trackedNodeIds: tracked,
            },
          };
        }),

      resetFlow: () =>
        set({
          learningFlow: defaultLearningFlow,
          currentSuggestion: null,
          suggestions: [],
        }),
    }),
    {
      name: "mentor-storage",
      partialize: (state) => ({
        preferences: state.preferences,
        history: state.history.slice(0, 10),
        learningFlow: state.learningFlow,
      }),
    },
  ),
);
