/**
 * Custom App Builder — TanStack Query hooks for CRUD operations.
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { appBuilderApi } from "../api/appBuilderApi";
import type {
  CreateAppRequest,
  UpdateAppRequest,
  PublishAppRequest,
} from "../types/appBuilder";

const KEYS = {
  all: ["custom-apps"] as const,
  list: () => [...KEYS.all, "list"] as const,
  detail: (id: number) => [...KEYS.all, "detail", id] as const,
  slug: (slug: string) => [...KEYS.all, "slug", slug] as const,
  suggest: (pipelineId: number) => [...KEYS.all, "suggest", pipelineId] as const,
};

// ─── Queries ──────────────────────────────────────────────────────

export function useCustomApps() {
  return useQuery({
    queryKey: KEYS.list(),
    queryFn: () => appBuilderApi.listApps(),
  });
}

export function useCustomApp(appId: number | undefined) {
  return useQuery({
    queryKey: KEYS.detail(appId!),
    queryFn: () => appBuilderApi.getApp(appId!),
    enabled: !!appId,
  });
}

export function useCheckSlug(slug: string) {
  return useQuery({
    queryKey: KEYS.slug(slug),
    queryFn: () => appBuilderApi.checkSlug(slug),
    enabled: slug.length > 0,
  });
}

export function useSuggestBlocks(pipelineId: number | undefined) {
  return useQuery({
    queryKey: KEYS.suggest(pipelineId!),
    queryFn: () => appBuilderApi.suggestBlocks(pipelineId!),
    enabled: !!pipelineId,
    staleTime: 5 * 60 * 1000, // suggestions don't change often
  });
}

// ─── Mutations ────────────────────────────────────────────────────

export function useCreateApp() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: CreateAppRequest) => appBuilderApi.createApp(data),
    onSuccess: () => qc.invalidateQueries({ queryKey: KEYS.list() }),
  });
}

export function useUpdateApp(appId: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: UpdateAppRequest) => appBuilderApi.updateApp(appId, data),
    onSuccess: (updated) => {
      qc.setQueryData(KEYS.detail(appId), updated);
      qc.invalidateQueries({ queryKey: KEYS.list() });
    },
  });
}

export function useDeleteApp() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (appId: number) => appBuilderApi.deleteApp(appId),
    onSuccess: () => qc.invalidateQueries({ queryKey: KEYS.list() }),
  });
}

export function usePublishApp(appId: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: PublishAppRequest) => appBuilderApi.publishApp(appId, data),
    onSuccess: (updated) => {
      qc.setQueryData(KEYS.detail(appId), updated);
      qc.invalidateQueries({ queryKey: KEYS.list() });
    },
  });
}
