/**
 * Zustand store for gamification state
 */

import { create } from "zustand";
import type { Badge } from "../../../types/api";

interface GamificationStore {
  xp: number;
  level: number;
  xpToNextLevel: number;
  progressPercent: number;
  badges: Badge[];

  // UI state
  showLevelUp: boolean;
  newLevel: number | null;
  recentXPGain: number | null;
  recentBadges: Badge[];

  // Actions
  setProfile: (profile: {
    xp: number;
    level: number;
    xp_to_next_level: number;
    progress_percent: number;
    badges: Badge[];
  }) => void;
  handleXPAward: (result: {
    xp_gained: number;
    total_xp: number;
    level: number;
    leveled_up: boolean;
    new_level?: number;
    new_badges: Badge[];
  }) => void;
  dismissLevelUp: () => void;
  clearRecentXP: () => void;
  clearRecentBadges: () => void;
}

export const useGamificationStore = create<GamificationStore>((set) => ({
  xp: 0,
  level: 1,
  xpToNextLevel: 100,
  progressPercent: 0,
  badges: [],

  showLevelUp: false,
  newLevel: null,
  recentXPGain: null,
  recentBadges: [],

  setProfile: (profile) =>
    set({
      xp: profile.xp,
      level: profile.level,
      xpToNextLevel: profile.xp_to_next_level,
      progressPercent: profile.progress_percent,
      badges: profile.badges,
    }),

  handleXPAward: (result) =>
    set((state) => ({
      xp: result.total_xp,
      level: result.level,
      recentXPGain: result.xp_gained,
      recentBadges: result.new_badges,
      showLevelUp: result.leveled_up,
      newLevel: result.new_level ?? null,
      badges: state.badges.map((b) => {
        const newBadge = result.new_badges.find(
          (nb) => nb.badge_id === b.badge_id,
        );
        return newBadge ? { ...b, is_earned: true, awarded_at: newBadge.awarded_at } : b;
      }),
    })),

  dismissLevelUp: () => set({ showLevelUp: false, newLevel: null }),
  clearRecentXP: () => set({ recentXPGain: null }),
  clearRecentBadges: () => set({ recentBadges: [] }),
}));
