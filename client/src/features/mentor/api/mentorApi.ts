/**
 * Mentor API Client
 *
 * Handles communication with backend mentor endpoints
 */

import axios from "axios";
import type {
  MentorSuggestion,
  MentorPreferences,
  DatasetInsight,
} from "../store/mentorStore";

const API_BASE = "/api/v1/mentor";

export interface MentorResponse {
  success: boolean;
  greeting?: string;
  suggestions: MentorSuggestion[];
  dataset_insights?: DatasetInsight;
  next_steps?: string[];
  context_summary?: string;
}

export interface MentorAnalysisRequest {
  nodes: Array<Record<string, unknown>>;
  edges: Array<Record<string, unknown>>;
  dataset_metadata?: Record<string, unknown>;
  current_context?: string;
  user_message?: string;
}

export interface TTSResponse {
  success: boolean;
  audio_url?: string;
  audio_base64?: string;
  duration_seconds?: number;
  cached: boolean;
  error?: string;
}

export interface PipelineStepGuide {
  model_type: string;
  steps: Array<{
    step: number;
    node_type: string;
    description: string;
  }>;
  explanation: string;
  estimated_time?: string;
}

class MentorAPI {
  /**
   * Get personalized greeting
   */
  async greetUser(
    userName: string,
    timeOfDay?: string,
  ): Promise<MentorResponse> {
    console.log("[mentorApi] greetUser called:", { userName, timeOfDay });
    const response = await axios.post<MentorResponse>(`${API_BASE}/greet`, {
      user_name: userName,
      time_of_day: timeOfDay,
    });
    console.log("[mentorApi] greetUser response:", response.data);
    return response.data;
  }

  /**
   * Analyze dataset and get insights
   */
  async analyzeDataset(
    request: MentorAnalysisRequest,
  ): Promise<MentorResponse> {
    const response = await axios.post<MentorResponse>(
      `${API_BASE}/analyze-dataset`,
      request,
    );
    return response.data;
  }

  /**
   * Analyze pipeline state and get suggestions
   */
  async analyzePipeline(
    request: MentorAnalysisRequest,
  ): Promise<MentorResponse> {
    const response = await axios.post<MentorResponse>(
      `${API_BASE}/analyze-pipeline`,
      request,
    );
    return response.data;
  }

  /**
   * Get explanation for execution error
   */
  async explainError(
    errorMessage: string,
    nodeType: string,
    nodeConfig: Record<string, unknown>,
    pipelineState?: Record<string, unknown>,
  ): Promise<MentorResponse> {
    const response = await axios.post<MentorResponse>(
      `${API_BASE}/explain-error`,
      {
        error_message: errorMessage,
        node_type: nodeType,
        node_config: nodeConfig,
        pipeline_state: pipelineState,
      },
    );
    return response.data;
  }

  /**
   * Generate speech audio from text
   */
  async generateSpeech(
    text: string,
    personality: string = "encouraging",
    voiceId?: string,
    cacheKey?: string,
  ): Promise<TTSResponse> {
    console.log("[mentorApi] generateSpeech called:", {
      text: text.substring(0, 50) + "...",
      personality,
      voiceId,
    });
    try {
      const response = await axios.post<TTSResponse>(
        `${API_BASE}/generate-speech`,
        {
          text,
          personality,
          voice_id: voiceId || undefined,
          cache_key: cacheKey,
        },
      );
      console.log("[mentorApi] generateSpeech response:", response.data);
      return response.data;
    } catch (error) {
      console.error("[mentorApi] generateSpeech error:", error);
      throw error;
    }
  }

  /**
   * Preview a voice with a short sample
   */
  async previewVoice(
    voiceId: string,
    text?: string,
  ): Promise<TTSResponse> {
    const response = await axios.post<TTSResponse>(
      `${API_BASE}/preview-voice`,
      {
        text: text || "",
        voice_id: voiceId,
      },
    );
    return response.data;
  }

  /**
   * Clone a voice from an audio file
   */
  async cloneVoice(
    displayName: string,
    audioFile: File,
  ): Promise<{ success: boolean; voice_id: string; display_name: string }> {
    const formData = new FormData();
    formData.append("file", audioFile);
    formData.append("display_name", displayName);

    const response = await axios.post<{
      success: boolean;
      voice_id: string;
      display_name: string;
    }>(`${API_BASE}/clone-voice`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 120000,
    });
    return response.data;
  }

  /**
   * Get user's mentor preferences
   */
  async getPreferences(): Promise<MentorPreferences> {
    const response = await axios.get<MentorPreferences>(
      `${API_BASE}/preferences`,
    );
    return response.data;
  }

  /**
   * Update user's mentor preferences
   */
  async updatePreferences(
    preferences: MentorPreferences,
  ): Promise<MentorPreferences> {
    const response = await axios.put<MentorPreferences>(
      `${API_BASE}/preferences`,
      preferences,
    );
    return response.data;
  }

  /**
   * Get step-by-step guide for a model type
   */
  async getModelGuide(modelType: string): Promise<PipelineStepGuide> {
    const response = await axios.post<PipelineStepGuide>(
      `${API_BASE}/get-guide/${modelType}`,
    );
    return response.data;
  }

  /**
   * Get detailed model introduction with interactive options
   */
  async getModelIntroduction(modelType: string): Promise<MentorResponse> {
    const response = await axios.post<MentorResponse>(
      `${API_BASE}/model-introduction/${modelType}`,
    );
    return response.data;
  }

  /**
   * Get dataset preparation guidance based on user's choice
   */
  async getDatasetGuidance(
    action: string,
    modelType: string,
    nextMessage: string,
  ): Promise<MentorResponse> {
    const response = await axios.post<MentorResponse>(
      `${API_BASE}/dataset-guidance`,
      {
        action,
        model_type: modelType,
        next_message: nextMessage,
      },
    );
    return response.data;
  }
}

export const mentorApi = new MentorAPI();
