/**
 * Gamification API client
 */

import api from "../../../lib/axios";
import type { GamificationProfile, AwardXPResponse } from "../../../types/api";

export const gamificationApi = {
  getProfile: async (): Promise<GamificationProfile> => {
    const { data } = await api.get("/gamification/profile");
    return data;
  },

  awardXP: async (
    action: string,
    context?: Record<string, unknown>,
  ): Promise<AwardXPResponse> => {
    const { data } = await api.post("/gamification/award-xp", {
      action,
      context,
    });
    return data;
  },

  getBadges: async () => {
    const { data } = await api.get("/gamification/badges");
    return data.badges;
  },
};
