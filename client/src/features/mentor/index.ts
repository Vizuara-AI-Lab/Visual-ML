/**
 * Mentor Feature Index
 *
 * Exports all mentor-related components, hooks, and utilities
 */

// Components
export { MentorAssistant } from "./components/MentorAssistant";
export { CharacterAvatar } from "./components/CharacterAvatar";
export { SuggestionCard } from "./components/SuggestionCard";
export { GuideCard } from "./components/GuideCard";

// Store
export { useMentorStore } from "./store/mentorStore";
export type {
  MentorSuggestion,
  MentorPreferences,
  MentorAction,
  DatasetInsight,
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
