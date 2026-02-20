/**
 * Learning Flow Hook - State Machine
 *
 * Watches playground state and advances the learning flow one step at a time.
 * Replaces useMentorContext's multiple ad-hoc useEffects with a deterministic
 * state machine that fires exactly one message per transition.
 *
 * Architecture:
 *   Effect 1 – Stage entry: fires when `stage` changes. Shows the message
 *              for the new stage and handles auto-advances (with optional delay).
 *   Effect 2 – Node guard: fires when `nodes` change. Checks whether the
 *              current stage's guard condition is met (e.g. a new node appeared
 *              or became configured) and advances the stage.
 *   Effect 3 – Execution guard: fires when `executionResult` changes. Handles
 *              guards that depend on pipeline execution (column info results,
 *              final run results, errors).
 *   Effect 4 – Audio gate: fires when `isSpeaking` changes. For stages that
 *              must wait for TTS to finish before advancing (e.g. algorithm
 *              intro), advances only after audio playback completes.
 */

import { useEffect, useRef } from "react";
import { usePlaygroundStore } from "../../../store/playgroundStore";
import {
  useMentorStore,
  type LearningStage,
  type MentorSuggestion,
} from "../store/mentorStore";
import { getAlgorithmConfig } from "../content/algorithmConfig";
import type { FlowMessage } from "../content/learningMessages";
import * as msg from "../content/learningMessages";

// ── Helpers ─────────────────────────────────────────────────────

/** Convert a FlowMessage to a MentorSuggestion the store can display. */
function toSuggestion(
  flowMsg: FlowMessage,
  actions: MentorSuggestion["actions"] = [],
  type: MentorSuggestion["type"] = "next_step",
  priority: MentorSuggestion["priority"] = "info",
): MentorSuggestion {
  return {
    id: `flow-${Date.now()}`,
    type,
    priority,
    title: flowMsg.title,
    message: flowMsg.displayText,
    voice_text:
      flowMsg.voiceText !== flowMsg.displayText
        ? flowMsg.voiceText
        : undefined,
    actions,
    dismissible: false,
  };
}

const DATA_SOURCE_TYPES = ["upload_file", "select_dataset"];

/** Stages where auto-advance should wait for TTS to finish. */
const AUDIO_GATED_STAGES: Set<LearningStage> = new Set([
  "algorithm_selected",
  "pipeline_executed",
]);

// ── Hook ────────────────────────────────────────────────────────

interface UseLearningFlowOptions {
  enabled?: boolean;
}

export function useLearningFlow({
  enabled = true,
}: UseLearningFlowOptions = {}) {
  // ── Subscriptions (select individual slices to minimise re-renders) ──

  const nodes = usePlaygroundStore((s) => s.nodes);
  const executionResult = usePlaygroundStore((s) => s.executionResult);

  const stage = useMentorStore((s) => s.learningFlow.stage);
  const selectedAlgorithm = useMentorStore(
    (s) => s.learningFlow.selectedAlgorithm,
  );
  const datasetInfo = useMentorStore((s) => s.learningFlow.datasetInfo);
  const columnInfoResults = useMentorStore(
    (s) => s.learningFlow.columnInfoResults,
  );
  const executionResults = useMentorStore(
    (s) => s.learningFlow.executionResults,
  );
  const trackedNodeIds = useMentorStore(
    (s) => s.learningFlow.trackedNodeIds,
  );
  const previousStage = useMentorStore(
    (s) => s.learningFlow.previousStage,
  );
  const prefsEnabled = useMentorStore((s) => s.preferences.enabled);
  const isSpeaking = useMentorStore((s) => s.isSpeaking);

  const show = useMentorStore((s) => s.showSuggestion);
  const setStage = useMentorStore((s) => s.setStage);
  const resetFlow = useMentorStore((s) => s.resetFlow);
  const setDatasetInfo = useMentorStore((s) => s.setDatasetInfo);
  const setColumnInfoResults = useMentorStore((s) => s.setColumnInfoResults);
  const setExecutionResults = useMentorStore((s) => s.setExecutionResults);
  const trackNodeId = useMentorStore((s) => s.trackNodeId);

  const active = enabled && prefsEnabled;

  // ── Refs ──

  /** Guards against showing the same stage message twice. */
  const lastShownStageRef = useRef<LearningStage | null>(null);

  /** Timer id for delayed auto-advances so we can clean up on unmount. */
  const autoAdvanceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );

  /** Tracks whether TTS started playing for audio-gated transitions. */
  const audioStartedRef = useRef(false);

  // ────────────────────────────────────────────────────────────────
  // Effect 1 – STAGE ENTRY
  //
  // Fires every time `stage` changes. Responsible for:
  //   a) Showing the message for the new stage
  //   b) Scheduling auto-advances (immediate or delayed)
  // ────────────────────────────────────────────────────────────────

  useEffect(() => {
    if (!active) return;
    if (stage === "welcome") return; // Welcome handled by MentorAssistant

    // Avoid showing the same message twice (e.g. on re-render without stage change)
    if (lastShownStageRef.current === stage) return;
    lastShownStageRef.current = stage;

    // Clear any pending auto-advance timer from a previous stage
    if (autoAdvanceTimerRef.current) {
      clearTimeout(autoAdvanceTimerRef.current);
      autoAdvanceTimerRef.current = null;
    }

    // Reset audio gate for new stages
    audioStartedRef.current = false;

    switch (stage) {
      // ── Algorithm intro ──────────────────────────────────────
      // Audio-gated: waits for TTS to finish (Effect 4).
      // Fallback timer in case TTS is disabled or never starts.
      case "algorithm_selected": {
        if (!selectedAlgorithm) break;
        show(toSuggestion(msg.algorithmSelectedMessage(selectedAlgorithm)));
        // Fallback: if TTS doesn't play, advance after 12s (reading time)
        autoAdvanceTimerRef.current = setTimeout(
          () => setStage("prompt_drag_dataset"),
          12000,
        );
        break;
      }

      // ── Drag dataset onto canvas ─────────────────────────────
      case "prompt_drag_dataset": {
        show(
          toSuggestion(msg.promptDragDatasetMessage(), [
            {
              label: "Add Dataset Node",
              type: "add_node",
              payload: { node_type: "select_dataset" },
            },
          ]),
        );
        break;
      }

      // ── Dataset node detected, awaiting configuration ────────
      case "dataset_node_added": {
        show(toSuggestion(msg.datasetNodeAddedMessage()));
        break;
      }

      // ── Dataset configured → tell user to add Column Info ────
      case "dataset_configured": {
        const info = datasetInfo;
        if (info) {
          show(
            toSuggestion(
              msg.datasetConfiguredMessage(
                info.filename,
                info.nRows,
                info.nCols,
              ),
              [
                {
                  label: "Add Column Info",
                  type: "add_node",
                  payload: { node_type: "column_info" },
                },
              ],
            ),
          );
        }
        setStage("prompt_column_info");
        break;
      }

      // ── Waiting for Column Info node ─────────────────────────
      case "prompt_column_info": {
        show(
          toSuggestion(msg.promptColumnInfoMessage(), [
            {
              label: "Add Column Info",
              type: "add_node",
              payload: { node_type: "column_info" },
            },
          ]),
        );
        break;
      }

      // ── Column Info connected → tell user to Run ─────────────
      case "column_info_added": {
        show(toSuggestion(msg.columnInfoAddedMessage()));
        setStage("prompt_run_column_info");
        break;
      }

      // ── Waiting for pipeline run with Column Info ─────────────
      case "prompt_run_column_info": {
        show(
          toSuggestion(msg.promptRunColumnInfoMessage(), [
            { label: "Run Pipeline", type: "execute", payload: {} },
          ]),
        );
        break;
      }

      // ── Column Info results analysed → decide next step ──────
      case "column_info_executed": {
        const cr = columnInfoResults;
        if (!cr) break;

        if (cr.totalMissing > 0) {
          show(
            toSuggestion(
              msg.columnInfoExecutedMissingMessage(
                cr.missingColumns,
                cr.totalMissing,
              ),
              [
                {
                  label: "Add Missing Value Handler",
                  type: "add_node",
                  payload: { node_type: "missing_value_handler" },
                },
              ],
              "warning",
              "warning",
            ),
          );
          setStage("prompt_missing_values");
        } else if (cr.categoricalColumns.length > 0) {
          show(
            toSuggestion(
              msg.columnInfoExecutedCategoricalMessage(cr.categoricalColumns),
              [
                {
                  label: "Add Encoding",
                  type: "add_node",
                  payload: { node_type: "encoding" },
                },
              ],
            ),
          );
          setStage("prompt_encoding");
        } else {
          show(
            toSuggestion(msg.columnInfoExecutedCleanMessage(), [
              {
                label: "Add Target & Split",
                type: "add_node",
                payload: { node_type: "split" },
              },
            ]),
          );
          setStage("prompt_split");
        }
        break;
      }

      // ── Waiting for Missing Value Handler node ────────────────
      case "prompt_missing_values": {
        break;
      }

      // ── MVH node added → tell user to configure it ────────────
      case "missing_values_added": {
        show(toSuggestion(msg.missingValuesAddedMessage()));
        // DON'T auto-advance. Node guard (Effect 2) checks isConfigured.
        break;
      }

      // ── MVH configured → decide encoding or split ─────────────
      case "missing_values_configured": {
        const cr = columnInfoResults;
        if (cr && cr.categoricalColumns.length > 0) {
          show(
            toSuggestion(
              msg.promptEncodingAfterMissingMessage(cr.categoricalColumns),
              [
                {
                  label: "Add Encoding",
                  type: "add_node",
                  payload: { node_type: "encoding" },
                },
              ],
            ),
          );
          setStage("prompt_encoding");
        } else {
          show(
            toSuggestion(msg.promptSplitMessage(), [
              {
                label: "Add Target & Split",
                type: "add_node",
                payload: { node_type: "split" },
              },
            ]),
          );
          setStage("prompt_split");
        }
        break;
      }

      // ── Waiting for Encoding node ─────────────────────────────
      case "prompt_encoding": {
        break;
      }

      // ── Encoding added → prompt split ─────────────────────────
      case "encoding_added": {
        show(
          toSuggestion(msg.encodingAddedMessage(), [
            {
              label: "Add Target & Split",
              type: "add_node",
              payload: { node_type: "split" },
            },
          ]),
        );
        setStage("prompt_split");
        break;
      }

      // ── Waiting for Split node ────────────────────────────────
      case "prompt_split": {
        show(
          toSuggestion(msg.promptSplitMessage(), [
            {
              label: "Add Target & Split",
              type: "add_node",
              payload: { node_type: "split" },
            },
          ]),
        );
        break;
      }

      // ── Split node added → tell user to configure it ──────────
      case "split_added": {
        show(toSuggestion(msg.splitNodeAddedMessage()));
        // DON'T auto-advance. Node guard (Effect 2) checks isConfigured.
        break;
      }

      // ── Split configured → prompt model ───────────────────────
      case "split_configured": {
        if (!selectedAlgorithm) break;
        const config = getAlgorithmConfig(selectedAlgorithm);
        show(
          toSuggestion(msg.splitConfiguredMessage(selectedAlgorithm), [
            {
              label: `Add ${config.displayName}`,
              type: "add_node",
              payload: { node_type: config.nodeType },
            },
          ]),
        );
        setStage("prompt_model");
        break;
      }

      // ── Waiting for model node ────────────────────────────────
      case "prompt_model": {
        if (!selectedAlgorithm) break;
        const config = getAlgorithmConfig(selectedAlgorithm);
        show(
          toSuggestion(msg.splitConfiguredMessage(selectedAlgorithm), [
            {
              label: `Add ${config.displayName}`,
              type: "add_node",
              payload: { node_type: config.nodeType },
            },
          ]),
        );
        break;
      }

      // ── Model added → prompt metrics ──────────────────────────
      case "model_added": {
        if (!selectedAlgorithm) break;
        const config = getAlgorithmConfig(selectedAlgorithm);
        const metricActions = config.metricNodes.map((nodeType) => ({
          label: `Add ${nodeType.replace(/_/g, " ")}`,
          type: "add_node" as const,
          payload: { node_type: nodeType },
        }));
        show(
          toSuggestion(msg.modelAddedMessage(selectedAlgorithm), metricActions),
        );
        setStage("prompt_metrics");
        break;
      }

      // ── Waiting for metric nodes ──────────────────────────────
      case "prompt_metrics": {
        if (!selectedAlgorithm) break;
        const config = getAlgorithmConfig(selectedAlgorithm);
        const metricActions = config.metricNodes.map((nodeType) => ({
          label: `Add ${nodeType.replace(/_/g, " ")}`,
          type: "add_node" as const,
          payload: { node_type: nodeType },
        }));
        show(
          toSuggestion(msg.modelAddedMessage(selectedAlgorithm), metricActions),
        );
        break;
      }

      // ── Metrics connected → prompt final run ──────────────────
      case "metrics_added": {
        show(
          toSuggestion(msg.metricsAddedMessage(), [
            { label: "Run Pipeline", type: "execute", payload: {} },
          ]),
        );
        setStage("prompt_final_run");
        break;
      }

      // ── Waiting for final pipeline execution ──────────────────
      case "prompt_final_run": {
        show(
          toSuggestion(msg.promptFinalRunMessage(), [
            { label: "Run Pipeline", type: "execute", payload: {} },
          ]),
        );
        break;
      }

      // ── Pipeline executed → show results ──────────────────────
      // Audio-gated: waits for TTS to finish (Effect 4).
      case "pipeline_executed": {
        if (!selectedAlgorithm) break;
        const results = executionResults || {};
        show(
          toSuggestion(
            msg.pipelineExecutedMessage(selectedAlgorithm, results),
          ),
        );
        // Fallback: if TTS doesn't play, advance after 15s
        autoAdvanceTimerRef.current = setTimeout(
          () => setStage("completed"),
          15000,
        );
        break;
      }

      // ── Completed ─────────────────────────────────────────────
      case "completed": {
        if (!selectedAlgorithm) break;
        show(
          toSuggestion(msg.completedMessage(selectedAlgorithm), [
            {
              label: "Start Over",
              type: "select_algorithm",
              payload: { action: "reset" },
            },
          ]),
        );
        break;
      }

      // ── Error ─────────────────────────────────────────────────
      case "error_occurred": {
        // On page refresh executionResult is null — recover to previous stage
        if (!executionResult?.error) {
          if (previousStage && previousStage !== "error_occurred") {
            setStage(previousStage);
          } else {
            resetFlow();
          }
          break;
        }

        const errorText = executionResult.error;
        const lower = errorText.toLowerCase();

        if (
          lower.includes("missing value") ||
          lower.includes("missing data") ||
          lower.includes("nan") ||
          lower.includes("null") ||
          lower.includes("isnull")
        ) {
          show(
            toSuggestion(
              msg.missingValueErrorMessage(errorText),
              [
                {
                  label: "Add Missing Value Handler",
                  type: "add_node",
                  payload: { node_type: "missing_value_handler" },
                },
              ],
              "warning",
              "warning",
            ),
          );
        } else if (
          lower.includes("could not convert string") ||
          lower.includes("categorical") ||
          lower.includes("object") ||
          lower.includes("dtype") ||
          lower.includes("string to float") ||
          lower.includes("invalid literal")
        ) {
          show(
            toSuggestion(
              msg.encodingErrorMessage(errorText),
              [
                {
                  label: "Add Encoding",
                  type: "add_node",
                  payload: { node_type: "encoding" },
                },
              ],
              "warning",
              "warning",
            ),
          );
        } else {
          show(
            toSuggestion(
              msg.errorOccurredMessage(errorText),
              [],
              "error_explanation",
              "critical",
            ),
          );
        }
        break;
      }

      default:
        break;
    }

    return () => {
      if (autoAdvanceTimerRef.current) {
        clearTimeout(autoAdvanceTimerRef.current);
        autoAdvanceTimerRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stage, active]);

  // ────────────────────────────────────────────────────────────────
  // Effect 2 – NODE GUARD
  //
  // Fires when the canvas nodes change. Checks whether the current
  // stage's guard condition is satisfied (a specific node type
  // appeared or became configured) and advances the stage.
  // ────────────────────────────────────────────────────────────────

  useEffect(() => {
    if (!active) return;

    switch (stage) {
      // ── Also detect dataset drag during algorithm intro ───────
      case "algorithm_selected":
      case "prompt_drag_dataset": {
        const dsNode = nodes.find((n) =>
          DATA_SOURCE_TYPES.includes(n.data.type),
        );
        if (dsNode) {
          // Cancel any pending auto-advance timer
          if (autoAdvanceTimerRef.current) {
            clearTimeout(autoAdvanceTimerRef.current);
            autoAdvanceTimerRef.current = null;
          }
          trackNodeId("dataset", dsNode.id);
          setStage("dataset_node_added");
        }
        break;
      }

      // ── Waiting for dataset configuration ─────────────────────
      case "dataset_node_added": {
        const dsNode = trackedNodeIds.dataset
          ? nodes.find((n) => n.id === trackedNodeIds.dataset)
          : nodes.find((n) => DATA_SOURCE_TYPES.includes(n.data.type));

        if (!dsNode) break;
        if (!trackedNodeIds.dataset) trackNodeId("dataset", dsNode.id);

        if (dsNode.data.isConfigured && dsNode.data.config?.dataset_id) {
          const cfg = dsNode.data.config;
          setDatasetInfo({
            filename: (cfg.filename as string) || "your dataset",
            nRows: (cfg.n_rows as number) || 0,
            nCols: (cfg.n_columns as number) || 0,
          });
          setStage("dataset_configured");
        }
        break;
      }

      // ── Waiting for Column Info node ──────────────────────────
      case "prompt_column_info": {
        const ciNode = nodes.find((n) => n.data.type === "column_info");
        if (ciNode) {
          trackNodeId("columnInfo", ciNode.id);
          setStage("column_info_added");
        }
        break;
      }

      // ── Waiting for Missing Value Handler node ────────────────
      case "prompt_missing_values": {
        const mvNode = nodes.find(
          (n) => n.data.type === "missing_value_handler",
        );
        if (mvNode) {
          trackNodeId("missingHandler", mvNode.id);
          setStage("missing_values_added");
        }
        break;
      }

      // ── Waiting for MVH to be configured ──────────────────────
      case "missing_values_added": {
        const mvNodeId = trackedNodeIds.missingHandler;
        if (!mvNodeId) break;
        const mvNode = nodes.find((n) => n.id === mvNodeId);
        if (mvNode?.data.isConfigured) {
          setStage("missing_values_configured");
        }
        break;
      }

      // ── Waiting for Encoding node ─────────────────────────────
      case "prompt_encoding": {
        const encNode = nodes.find((n) => n.data.type === "encoding");
        if (encNode) {
          trackNodeId("encoding", encNode.id);
          setStage("encoding_added");
        }
        break;
      }

      // ── Waiting for Split node ────────────────────────────────
      case "prompt_split": {
        const splitNode = nodes.find((n) => n.data.type === "split");
        if (splitNode) {
          trackNodeId("split", splitNode.id);
          setStage("split_added");
        }
        break;
      }

      // ── Waiting for Split to be configured ────────────────────
      case "split_added": {
        const splitNodeId = trackedNodeIds.split;
        if (!splitNodeId) break;
        const splitNode = nodes.find((n) => n.id === splitNodeId);
        if (splitNode?.data.isConfigured) {
          setStage("split_configured");
        }
        break;
      }

      // ── Waiting for model node ────────────────────────────────
      case "prompt_model": {
        if (!selectedAlgorithm) break;
        const config = getAlgorithmConfig(selectedAlgorithm);
        const modelNode = nodes.find(
          (n) => n.data.type === config.nodeType,
        );
        if (modelNode) {
          trackNodeId("model", modelNode.id);
          setStage("model_added");
        }
        break;
      }

      // ── Waiting for metric nodes ──────────────────────────────
      case "prompt_metrics": {
        if (!selectedAlgorithm) break;
        const config = getAlgorithmConfig(selectedAlgorithm);
        const metricNode = nodes.find((n) =>
          config.metricNodes.includes(n.data.type),
        );
        if (metricNode) {
          trackNodeId("metrics", metricNode.id);
          setStage("metrics_added");
        }
        break;
      }

      default:
        break;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, stage, active]);

  // ────────────────────────────────────────────────────────────────
  // Effect 3 – EXECUTION GUARD
  //
  // Fires when executionResult changes. Handles:
  //   - Column Info results analysis
  //   - Final pipeline completion
  //   - Execution errors
  // ────────────────────────────────────────────────────────────────

  useEffect(() => {
    if (!active || !executionResult) return;

    // ── Error detection (from any stage) ────────────────────────
    if (
      !executionResult.success &&
      executionResult.error &&
      stage !== "error_occurred" &&
      stage !== "welcome"
    ) {
      setStage("error_occurred");
      return;
    }

    if (!executionResult.success) return;

    switch (stage) {
      // ── Column Info execution results ─────────────────────────
      case "prompt_run_column_info": {
        if (!executionResult.nodeResults) break;

        const ciNodeId = trackedNodeIds.columnInfo;
        if (!ciNodeId) break;

        const result = executionResult.nodeResults[ciNodeId];
        if (!result?.success || !result?.output) break;

        const output = result.output as {
          column_info?: Array<{
            column: string;
            dtype: string;
            missing?: number;
            unique?: number;
          }>;
        };
        if (!output.column_info || output.column_info.length === 0) break;

        // Analyse the column info
        const colInfo = output.column_info;
        const missingCols = colInfo.filter((c) => (c.missing ?? 0) > 0);
        const categoricalCols = colInfo.filter(
          (c) =>
            c.dtype === "object" ||
            c.dtype === "category" ||
            c.dtype === "string",
        );
        const numericCols = colInfo.filter(
          (c) =>
            c.dtype !== "object" &&
            c.dtype !== "category" &&
            c.dtype !== "string",
        );
        const totalMissing = missingCols.reduce(
          (sum, c) => sum + (c.missing ?? 0),
          0,
        );

        setColumnInfoResults({
          missingColumns: missingCols.map((c) => ({
            column: c.column,
            count: c.missing ?? 0,
          })),
          categoricalColumns: categoricalCols.map((c) => c.column),
          numericColumns: numericCols.map((c) => c.column),
          totalMissing,
        });

        setStage("column_info_executed");
        break;
      }

      // ── Final pipeline run ────────────────────────────────────
      case "prompt_final_run": {
        if (!executionResult.metrics) break;

        const results: Record<string, number> = {};
        for (const [key, val] of Object.entries(executionResult.metrics)) {
          if (typeof val === "number") {
            results[key] = val;
          }
        }

        setExecutionResults(results);
        setStage("pipeline_executed");
        break;
      }

      default:
        break;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [executionResult, stage, active]);

  // ────────────────────────────────────────────────────────────────
  // Effect 4 – AUDIO GATE
  //
  // For stages in AUDIO_GATED_STAGES, waits for TTS playback to
  // finish before auto-advancing. This prevents speech from being
  // cut off mid-sentence when the next message appears.
  // ────────────────────────────────────────────────────────────────

  useEffect(() => {
    if (!active) return;
    if (!AUDIO_GATED_STAGES.has(stage)) return;

    if (isSpeaking) {
      // TTS started → mark it so we know to advance when it stops
      audioStartedRef.current = true;
      return;
    }

    // isSpeaking is false — only advance if audio actually played and finished
    if (!audioStartedRef.current) return;
    audioStartedRef.current = false;

    // Clear the fallback timer since audio completed naturally
    if (autoAdvanceTimerRef.current) {
      clearTimeout(autoAdvanceTimerRef.current);
      autoAdvanceTimerRef.current = null;
    }

    switch (stage) {
      case "algorithm_selected":
        setStage("prompt_drag_dataset");
        break;
      case "pipeline_executed":
        setStage("completed");
        break;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isSpeaking, stage, active]);
}
