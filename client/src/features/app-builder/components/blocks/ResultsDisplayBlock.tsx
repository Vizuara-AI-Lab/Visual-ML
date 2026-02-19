/**
 * Results Display Block — Rich interactive renderer for pipeline results.
 * Shows ALL pipeline nodes (including preprocessing) with interactive explorers.
 * Uses framer-motion for staggered entrance animations.
 */

import { motion, AnimatePresence } from "framer-motion";
import {
  BarChart3,
  CheckCircle2,
  XCircle,
  Clock,
  Sparkles,
  ChevronDown,
} from "lucide-react";
import type { BlockRenderProps } from "../BlockRenderer";
import type { ResultsDisplayConfig } from "../../types/appBuilder";
import NodeExplorerPanel from "../NodeExplorerPanel";
import ExecutionTimeline, { SuccessBanner } from "../ExecutionTimeline";

// ─── Types ───────────────────────────────────────────────────────

interface NodeResult {
  node_type: string;
  execution_time_ms?: number;
  success?: boolean;
  error?: string | null;
  [key: string]: unknown;
}

// Node types that are purely internal / no useful output for end users
const SKIP_NODE_TYPES = new Set(["upload_file", "preprocess"]);

// ─── Main Component ──────────────────────────────────────────────

export default function ResultsDisplayBlock({
  block,
  mode,
  results,
  isExecuting,
  theme,
}: BlockRenderProps) {
  const config = block.config as ResultsDisplayConfig;
  const primaryColor = theme?.primaryColor ?? "#6366f1";

  // ── Edit / Preview placeholder ──
  if (mode !== "live") {
    return (
      <div className="bg-white rounded-xl border p-6">
        <h3 className="text-sm font-medium text-gray-700 mb-3">{config.title}</h3>
        <div className="bg-gradient-to-br from-indigo-50/50 to-purple-50/50 rounded-xl p-10 text-center border border-dashed border-indigo-200">
          <div className="w-12 h-12 mx-auto mb-3 rounded-xl bg-gradient-to-br from-indigo-100 to-purple-100 flex items-center justify-center">
            <BarChart3 className="h-6 w-6 text-indigo-400" />
          </div>
          <p className="text-sm font-medium text-gray-500">Interactive results will appear here</p>
          <p className="text-xs text-gray-400 mt-1">Dataset explorer, model insights, quizzes & more</p>
        </div>
      </div>
    );
  }

  // ── Executing — show timeline ──
  if (isExecuting) {
    return (
      <div className="space-y-4">
        {config.title && (
          <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-indigo-400" />
            {config.title}
          </h3>
        )}
        <ExecutionTimeline isExecuting={true} primaryColor={primaryColor} />
      </div>
    );
  }

  // ── No results yet (idle) ──
  if (!results) {
    return (
      <div className="bg-white rounded-xl border p-6">
        {config.title && <h3 className="text-sm font-medium text-gray-700 mb-3">{config.title}</h3>}
        <div className="bg-gradient-to-br from-indigo-50/30 to-purple-50/30 rounded-xl p-10 text-center border border-dashed border-indigo-100">
          <BarChart3 className="h-8 w-8 mx-auto text-gray-300 mb-2" />
          <p className="text-sm text-gray-400">Results will appear here after execution</p>
        </div>
      </div>
    );
  }

  // ── Error ──
  if (!results.success) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-xl border border-red-200 p-6"
      >
        <div className="flex items-center gap-2 mb-3">
          <XCircle className="h-5 w-5 text-red-500" />
          <h3 className="text-sm font-semibold text-red-700">{config.title || "Execution Failed"}</h3>
        </div>
        <div className="bg-red-50 rounded-lg p-4">
          <p className="text-sm text-red-600">{results.error || "Pipeline execution failed"}</p>
        </div>
      </motion.div>
    );
  }

  // ── Success — extract node results ──
  const allResults = (results.results ?? {}) as Record<string, unknown>;
  const nodeResultsArray = (allResults.results ?? []) as NodeResult[];
  const executionTimeMs = (allResults.execution_time_ms as number | undefined) ?? results.execution_time_ms;
  const nodesExecuted = nodeResultsArray.length;

  // Filter nodes we can actually show
  const visibleResults = nodeResultsArray.filter(
    (r) => !SKIP_NODE_TYPES.has(r.node_type)
  );

  return (
    <div className="space-y-4">
      {/* Title + success banner */}
      <div className="flex items-center justify-between">
        {config.title && (
          <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-green-500" />
            {config.title}
          </h3>
        )}
        {executionTimeMs != null && (
          <span className="flex items-center gap-1 text-xs text-gray-400">
            <Clock className="h-3 w-3" />
            {executionTimeMs < 1000 ? `${Math.round(executionTimeMs)}ms` : `${(executionTimeMs / 1000).toFixed(1)}s`}
          </span>
        )}
      </div>

      {/* Success banner */}
      <SuccessBanner executionTimeMs={executionTimeMs} nodeCount={nodesExecuted} />

      {/* Empty state */}
      {visibleResults.length === 0 && (
        <div className="bg-green-50 border border-green-200 rounded-xl p-8 text-center">
          <CheckCircle2 className="h-8 w-8 mx-auto text-green-400 mb-2" />
          <p className="text-sm font-medium text-green-700">Pipeline completed successfully</p>
          <p className="text-xs text-green-500 mt-1">No viewable output was generated</p>
        </div>
      )}

      {/* Staggered node result cards */}
      <AnimatePresence>
        <motion.div className="space-y-3">
          {visibleResults.map((nodeResult, idx) => (
            <motion.div
              key={`${nodeResult.node_type}-${idx}`}
              initial={{ opacity: 0, y: 20, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{
                delay: idx * 0.12,
                duration: 0.4,
                ease: [0.25, 0.46, 0.45, 0.94],
              }}
            >
              <NodeExplorerPanel result={nodeResult} primaryColor={primaryColor} />
            </motion.div>
          ))}
        </motion.div>
      </AnimatePresence>

      {/* Scroll hint when many results */}
      {visibleResults.length > 3 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: visibleResults.length * 0.12 + 0.3 }}
          className="flex items-center justify-center gap-2 text-xs text-indigo-400 pt-2"
        >
          <ChevronDown className="h-3 w-3" />
          <span>Scroll to explore all {visibleResults.length} results</span>
        </motion.div>
      )}
    </div>
  );
}
