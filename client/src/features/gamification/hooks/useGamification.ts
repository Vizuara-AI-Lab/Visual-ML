/**
 * TanStack Query hooks for gamification data
 */

import { useQuery, useMutation } from "@tanstack/react-query";
import { gamificationApi } from "../api/gamificationApi";
import { useGamificationStore } from "../store/gamificationStore";
import { useEffect } from "react";

const KEYS = {
  all: ["gamification"] as const,
  profile: () => [...KEYS.all, "profile"] as const,
  badges: () => [...KEYS.all, "badges"] as const,
};

export function useGamificationProfile() {
  const setProfile = useGamificationStore((s) => s.setProfile);

  const query = useQuery({
    queryKey: KEYS.profile(),
    queryFn: () => gamificationApi.getProfile(),
    staleTime: 30_000,
  });

  useEffect(() => {
    if (query.data) {
      setProfile(query.data);
    }
  }, [query.data, setProfile]);

  return query;
}

export function useGamificationBadges() {
  return useQuery({
    queryKey: KEYS.badges(),
    queryFn: () => gamificationApi.getBadges(),
    staleTime: 60_000,
  });
}

export function useAwardXP() {
  const handleXPAward = useGamificationStore((s) => s.handleXPAward);

  return useMutation({
    mutationFn: ({
      action,
      context,
    }: {
      action: string;
      context?: Record<string, unknown>;
    }) => gamificationApi.awardXP(action, context),
    onSuccess: (data) => {
      handleXPAward(data);
    },
  });
}
