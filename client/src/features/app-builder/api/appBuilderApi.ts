/**
 * Custom App Builder — API Client
 */

import { apiClient } from "../../../lib/axios";
import type {
  CustomApp,
  CreateAppRequest,
  UpdateAppRequest,
  PublishAppRequest,
  PublicApp,
  ExecuteAppRequest,
  ExecuteAppResponse,
  SlugCheckResponse,
  SuggestedBlocksResponse,
} from "../types/appBuilder";

const BASE = "/custom-apps";

class AppBuilderAPI {
  // ─── Authenticated CRUD ─────────────────────────────────────────

  async createApp(data: CreateAppRequest): Promise<CustomApp> {
    const res = await apiClient.post<CustomApp>(BASE, data);
    return res.data;
  }

  async listApps(): Promise<CustomApp[]> {
    const res = await apiClient.get<CustomApp[]>(BASE);
    return res.data;
  }

  async getApp(appId: number): Promise<CustomApp> {
    const res = await apiClient.get<CustomApp>(`${BASE}/${appId}`);
    return res.data;
  }

  async updateApp(appId: number, data: UpdateAppRequest): Promise<CustomApp> {
    const res = await apiClient.put<CustomApp>(`${BASE}/${appId}`, data);
    return res.data;
  }

  async deleteApp(appId: number): Promise<void> {
    await apiClient.delete(`${BASE}/${appId}`);
  }

  async publishApp(appId: number, data: PublishAppRequest): Promise<CustomApp> {
    const res = await apiClient.post<CustomApp>(`${BASE}/${appId}/publish`, data);
    return res.data;
  }

  async checkSlug(slug: string): Promise<SlugCheckResponse> {
    const res = await apiClient.get<SlugCheckResponse>(`${BASE}/check-slug/${slug}`);
    return res.data;
  }

  // ─── Pipeline Suggestions ────────────────────────────────────────

  async suggestBlocks(pipelineId: number): Promise<SuggestedBlocksResponse> {
    const res = await apiClient.get<SuggestedBlocksResponse>(
      `${BASE}/pipeline/${pipelineId}/suggest-blocks`,
    );
    return res.data;
  }

  // ─── Public (no auth) ──────────────────────────────────────────

  async getPublicApp(slug: string): Promise<PublicApp> {
    const res = await apiClient.get<PublicApp>(`${BASE}/public/${slug}`);
    return res.data;
  }

  async executePublicApp(slug: string, data: ExecuteAppRequest): Promise<ExecuteAppResponse> {
    const res = await apiClient.post<ExecuteAppResponse>(`${BASE}/public/${slug}/execute`, data);
    return res.data;
  }
}

export const appBuilderApi = new AppBuilderAPI();
