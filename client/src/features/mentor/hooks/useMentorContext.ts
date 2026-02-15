import { useEffect, useRef } from "react";
import { usePlaygroundStore } from "../../../store/playgroundStore";
import { useMentorStore } from "../store/mentorStore";
import { mentorApi } from "../api/mentorApi";

interface UseMentorContextOptions {
  /** Enable auto-triggering of mentor suggestions */
  enabled?: boolean;
  /** Debounce delay in milliseconds */
  debounceMs?: number;
}

/**
 * Hook that watches playground state changes and automatically triggers
 * relevant mentor suggestions based on user actions.
 */
export function useMentorContext(options: UseMentorContextOptions = {}) {
  const { enabled = true, debounceMs = 1000 } = options;

  const { nodes, edges, executionResult } = usePlaygroundStore();
  const { preferences, showSuggestion, queueAudio } = useMentorStore();

  const previousNodesCountRef = useRef(nodes.length);
  const previousExecutionResultRef = useRef(executionResult);
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Auto-analyze dataset when upload node is added and configured
  useEffect(() => {
    if (!enabled || !preferences.autoAnalyze) return;

    const uploadNodes = nodes.filter(
      (node) =>
        node.data.type === "upload_dataset" &&
        node.data.isConfigured &&
        node.data.dataset_id,
    );

    uploadNodes.forEach(async (node) => {
      const datasetId = node.data.dataset_id;
      if (!datasetId) return;

      try {
        const insights = await mentorApi.analyzeDataset(datasetId);

        if (insights.suggestions.length > 0) {
          insights.suggestions.forEach((suggestion) => {
            showSuggestion(suggestion);

            // Queue audio if voice mode is enabled
            if (preferences.voiceMode === "always" && suggestion.message) {
              mentorApi.generateSpeech(suggestion.message).then((audioData) => {
                if (audioData.audio_base64) {
                  queueAudio(audioData.audio_base64);
                }
              });
            }
          });
        }
      } catch (error) {
        console.error("Failed to analyze dataset:", error);
      }
    });
  }, [nodes, enabled, preferences.autoAnalyze, preferences.voiceMode]);

  // Detect when new nodes are added and provide contextual suggestions
  useEffect(() => {
    if (!enabled) return;

    const currentCount = nodes.length;
    const previousCount = previousNodesCountRef.current;

    if (currentCount > previousCount && currentCount > 0) {
      // Clear existing debounce timer
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }

      // Debounce pipeline analysis
      debounceTimerRef.current = setTimeout(async () => {
        try {
          const pipelineData = {
            nodes: nodes.map((n) => ({
              id: n.id,
              type: n.data.type,
              config: n.data,
              position: n.position,
            })),
            edges: edges.map((e) => ({
              id: e.id,
              source: e.source,
              target: e.target,
            })),
          };

          const analysis = await mentorApi.analyzePipeline(pipelineData);

          if (analysis.suggestions.length > 0) {
            // Only show the most important suggestions to avoid overwhelming
            const topSuggestions = analysis.suggestions
              .sort((a, b) => {
                const priorityOrder = { critical: 3, warning: 2, info: 1 };
                return (
                  (priorityOrder[b.priority] || 0) -
                  (priorityOrder[a.priority] || 0)
                );
              })
              .slice(0, 2);

            topSuggestions.forEach((suggestion) => {
              showSuggestion(suggestion);
            });
          }
        } catch (error) {
          console.error("Failed to analyze pipeline:", error);
        }
      }, debounceMs);
    }

    previousNodesCountRef.current = currentCount;

    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [nodes, edges, enabled, debounceMs]);

  // Detect execution errors and provide explanations
  useEffect(() => {
    if (!enabled) return;

    const currentResult = executionResult;
    const previousResult = previousExecutionResultRef.current;

    // Check if execution just completed with errors
    if (
      currentResult &&
      currentResult !== previousResult &&
      currentResult.status === "error" &&
      currentResult.error
    ) {
      (async () => {
        try {
          const explanation = await mentorApi.explainError(
            currentResult.error || "Unknown error occurred",
            currentResult.failedNodeId || undefined,
          );

          showSuggestion(explanation);

          // Always use voice for error explanations if voice is enabled
          if (preferences.voiceMode !== "never" && explanation.message) {
            const audioData = await mentorApi.generateSpeech(
              explanation.message,
            );
            if (audioData.audio_base64) {
              queueAudio(audioData.audio_base64);
            }
          }
        } catch (error) {
          console.error("Failed to explain error:", error);
        }
      })();
    }

    previousExecutionResultRef.current = currentResult;
  }, [executionResult, enabled, preferences.voiceMode]);

  // Detect when user might be stuck (no changes for a while)
  useEffect(() => {
    if (!enabled || !preferences.showTips) return;

    const inactivityTimeout = setTimeout(() => {
      if (nodes.length > 0 && nodes.length < 3) {
        showSuggestion({
          id: `inactivity-${Date.now()}`,
          type: "tip",
          priority: "info",
          message:
            "Need help building your pipeline? I can suggest next steps based on what you've started!",
          timestamp: new Date().toISOString(),
          actions: [
            {
              label: "Get Suggestions",
              type: "analyze_pipeline",
            },
          ],
        });
      }
    }, 60000); // 1 minute of inactivity

    return () => clearTimeout(inactivityTimeout);
  }, [nodes, enabled, preferences.showTips]);
}
