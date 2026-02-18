/**
 * Custom App Builder — Hooks for public app view + execution.
 */

import { useQuery, useMutation } from "@tanstack/react-query";
import { appBuilderApi } from "../api/appBuilderApi";
import type { ExecuteAppRequest } from "../types/appBuilder";

export function usePublicApp(slug: string | undefined) {
  return useQuery({
    queryKey: ["public-app", slug],
    queryFn: () => appBuilderApi.getPublicApp(slug!),
    enabled: !!slug,
    retry: false,
  });
}

export function useExecutePublicApp(slug: string) {
  return useMutation({
    mutationFn: (data: ExecuteAppRequest) =>
      appBuilderApi.executePublicApp(slug, data),
  });
}
