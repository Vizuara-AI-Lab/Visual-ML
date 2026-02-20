/**
 * Mentor Context Hook - Thin Wrapper
 *
 * Delegates to useLearningFlow for the structured learning state machine.
 * Kept as a wrapper to preserve the existing import interface used by
 * PlayGround.tsx and other consumers.
 */

import { useLearningFlow } from "./useLearningFlow";

interface UseMentorContextOptions {
  enabled?: boolean;
}

export function useMentorContext(options: UseMentorContextOptions = {}) {
  useLearningFlow({ enabled: options.enabled ?? true });
}
