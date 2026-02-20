/**
 * Mentor Feature Index
 *
 * Exports all mentor-related components, hooks, and utilities
 */

// Components
export { MentorAssistant } from "./components/MentorAssistant";
export { CharacterAvatar } from "./components/CharacterAvatar";
export { SuggestionCard } from "./components/SuggestionCard";

// Store
export { useMentorStore } from "./store/mentorStore";
export type {
  MentorSuggestion,
  MentorPreferences,
  MentorAction,
  DatasetInsight,
  LearningStage,
} from "./store/mentorStore";

// API
export { mentorApi } from "./api/mentorApi";
export type {
  MentorResponse,
  MentorAnalysisRequest,
  TTSResponse,
  PipelineStepGuide,
} from "./api/mentorApi";

// Hooks
export { useAudioPlayer } from "./hooks/useAudioPlayer";
export { useMentorContext } from "./hooks/useMentorContext";
export { useLearningFlow } from "./hooks/useLearningFlow";

// Content
export { ALGORITHM_CONFIG, getAlgorithmConfig } from "./content/algorithmConfig";
