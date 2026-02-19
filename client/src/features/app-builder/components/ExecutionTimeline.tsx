/**
 * ExecutionTimeline — Animated pipeline execution timeline.
 * Shows staged steps during execution, transforms to real node results after.
 */

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Database,
  Cpu,
  BarChart3,
  CheckCircle2,
  Loader2,
  Sparkles,
  GitBranch,
  BrainCircuit,
  FlaskConical,
  ChevronDown,
  Clock,
  Zap,
} from "lucide-react";

// ─── Types ───────────────────────────────────────────────────────

interface ExecutionTimelineProps {
  isExecuting: boolean;
  executionTimeMs?: number;
  primaryColor?: string;
}

// ─── Generic Stage Definitions (shown during execution) ──────────

interface Stage {
  id: string;
  label: string;
  detail: string;
  icon: React.ElementType;
  color: string;
}

const PIPELINE_STAGES: Stage[] = [
  { id: "load",    label: "Loading Data",      detail: "Reading dataset from storage",         icon: Database,     color: "#3b82f6" },
  { id: "prep",    label: "Preprocessing",     detail: "Cleaning and transforming data",       icon: FlaskConical, color: "#f59e0b" },
  { id: "model",   label: "Training Model",    detail: "Fitting ML algorithm to your data",   icon: BrainCircuit, color: "#8b5cf6" },
  { id: "split",   label: "Splitting Dataset", detail: "Creating train / test sets",           icon: GitBranch,    color: "#06b6d4" },
  { id: "eval",    label: "Evaluating",        detail: "Computing performance metrics",        icon: BarChart3,    color: "#10b981" },
  { id: "done",    label: "Finishing up",      detail: "Packaging results for you",            icon: Sparkles,     color: "#6366f1" },
];

// ─── Main Component ──────────────────────────────────────────────

export default function ExecutionTimeline({
  isExecuting,
  executionTimeMs,
  primaryColor = "#6366f1",
}: ExecutionTimelineProps) {
  const [activeStageIdx, setActiveStageIdx] = useState(0);
  const [completedStages, setCompletedStages] = useState<Set<number>>(new Set());

  // Cycle through stages while executing
  useEffect(() => {
    if (!isExecuting) {
      setActiveStageIdx(0);
      setCompletedStages(new Set());
      return;
    }

    // Start cycling through stages
    const STAGE_DURATION = 1200; // ms per stage
    let currentIdx = 0;

    const interval = setInterval(() => {
      setCompletedStages((prev) => new Set([...prev, currentIdx]));
      currentIdx = Math.min(currentIdx + 1, PIPELINE_STAGES.length - 1);
      setActiveStageIdx(currentIdx);
    }, STAGE_DURATION);

    return () => clearInterval(interval);
  }, [isExecuting]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -16 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="bg-white rounded-2xl border border-indigo-100 overflow-hidden shadow-lg"
    >
      {/* Header */}
      <div
        className="px-6 py-4 flex items-center gap-3"
        style={{ background: `linear-gradient(135deg, ${primaryColor}15, ${primaryColor}08)` }}
      >
        <div
          className="p-2 rounded-xl"
          style={{ backgroundColor: `${primaryColor}20` }}
        >
          {isExecuting ? (
            <Loader2 className="h-5 w-5 animate-spin" style={{ color: primaryColor }} />
          ) : (
            <CheckCircle2 className="h-5 w-5 text-green-500" />
          )}
        </div>
        <div>
          <h3 className="text-sm font-bold text-gray-800">
            {isExecuting ? "Running Pipeline…" : "Pipeline Complete!"}
          </h3>
          <p className="text-xs text-gray-500">
            {isExecuting
              ? "Your data is flowing through the ML pipeline"
              : executionTimeMs != null
                ? `Completed in ${executionTimeMs < 1000 ? `${executionTimeMs}ms` : `${(executionTimeMs / 1000).toFixed(1)}s`}`
                : "Results are ready below"}
          </p>
        </div>
        {!isExecuting && executionTimeMs != null && (
          <div className="ml-auto flex items-center gap-1 text-xs text-green-600 bg-green-50 px-3 py-1 rounded-full border border-green-200">
            <Clock className="h-3 w-3" />
            <span className="font-semibold">
              {executionTimeMs < 1000 ? `${executionTimeMs}ms` : `${(executionTimeMs / 1000).toFixed(1)}s`}
            </span>
          </div>
        )}
      </div>

      {/* Timeline */}
      <div className="px-6 py-5">
        <div className="relative">
          {/* Vertical track */}
          <div className="absolute left-[19px] top-4 bottom-4 w-0.5 bg-gray-100" />

          <div className="space-y-3">
            {PIPELINE_STAGES.map((stage, idx) => {
              const isActive = isExecuting && idx === activeStageIdx;
              const isDone = completedStages.has(idx) && !isActive;
              const isPending = !isActive && !isDone;
              const Icon = stage.icon;

              return (
                <motion.div
                  key={stage.id}
                  initial={{ opacity: 0, x: -12 }}
                  animate={{ opacity: isPending && isExecuting ? 0.4 : 1, x: 0 }}
                  transition={{ delay: idx * 0.08, duration: 0.3 }}
                  className="flex items-center gap-4 relative"
                >
                  {/* Icon bubble */}
                  <div
                    className={`relative z-10 flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center transition-all duration-500 ${
                      isActive
                        ? "shadow-lg ring-4 ring-offset-1"
                        : isDone
                          ? "shadow-sm"
                          : "border-2 border-gray-200 bg-white"
                    }`}
                    style={
                      isActive
                        ? {
                            backgroundColor: `${stage.color}20`,
                            ringColor: `${stage.color}30`,
                          }
                        : isDone
                          ? { backgroundColor: "#f0fdf4", borderColor: "#86efac" }
                          : undefined
                    }
                  >
                    {isDone ? (
                      <CheckCircle2 className="h-4 w-4 text-green-500" />
                    ) : isActive ? (
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                      >
                        <Icon className="h-4 w-4" style={{ color: stage.color }} />
                      </motion.div>
                    ) : (
                      <Icon className="h-4 w-4 text-gray-300" />
                    )}

                    {/* Active pulse ring */}
                    {isActive && (
                      <motion.div
                        className="absolute inset-0 rounded-full"
                        style={{ backgroundColor: `${stage.color}20` }}
                        animate={{ scale: [1, 1.6, 1], opacity: [0.8, 0, 0.8] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                      />
                    )}
                  </div>

                  {/* Text */}
                  <div className="flex-1 min-w-0">
                    <p
                      className={`text-sm font-semibold transition-colors ${
                        isActive ? "text-gray-900" : isDone ? "text-gray-700" : "text-gray-400"
                      }`}
                    >
                      {stage.label}
                    </p>
                    {isActive && (
                      <motion.p
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="text-xs text-gray-500 mt-0.5"
                      >
                        {stage.detail}
                      </motion.p>
                    )}
                    {isDone && (
                      <motion.p
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="text-xs text-green-600 mt-0.5"
                      >
                        Done
                      </motion.p>
                    )}
                  </div>

                  {/* Active badge */}
                  {isActive && (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="flex items-center gap-1 text-xs font-semibold px-2.5 py-1 rounded-full"
                      style={{ backgroundColor: `${stage.color}15`, color: stage.color }}
                    >
                      <Loader2 className="h-2.5 w-2.5 animate-spin" />
                      Running
                    </motion.div>
                  )}
                </motion.div>
              );
            })}
          </div>
        </div>

        {/* Animated progress bar */}
        {isExecuting && (
          <div className="mt-5 pt-4 border-t">
            <div className="flex items-center justify-between text-xs text-gray-500 mb-1.5">
              <span>Processing…</span>
              <span>{Math.round(((activeStageIdx + 1) / PIPELINE_STAGES.length) * 100)}%</span>
            </div>
            <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
              <motion.div
                className="h-full rounded-full"
                style={{ background: `linear-gradient(90deg, ${primaryColor}, ${primaryColor}cc)` }}
                animate={{ width: `${((activeStageIdx + 1) / PIPELINE_STAGES.length) * 100}%` }}
                transition={{ duration: 0.8, ease: "easeOut" }}
              />
            </div>
          </div>
        )}

        {/* Done state */}
        {!isExecuting && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="mt-5 pt-4 border-t flex items-center justify-center gap-2"
          >
            <div className="flex items-center gap-2 text-sm font-medium text-green-600 bg-green-50 px-4 py-2 rounded-full border border-green-200">
              <Zap className="h-4 w-4" />
              Scroll down to explore your results
            </div>
            <motion.div
              animate={{ y: [0, 4, 0] }}
              transition={{ duration: 1.2, repeat: Infinity }}
            >
              <ChevronDown className="h-4 w-4 text-green-500" />
            </motion.div>
          </motion.div>
        )}
      </div>

      {/* Animated bottom accent */}
      {isExecuting && (
        <motion.div
          className="h-1"
          style={{ background: `linear-gradient(90deg, transparent, ${primaryColor}, transparent)` }}
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
      )}
    </motion.div>
  );
}

// ─── Compact Executing Indicator (used inline in submit button area) ──

export function ExecutingIndicator({ primaryColor = "#6366f1" }: { primaryColor?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="flex items-center justify-center gap-3 py-3 px-5 rounded-xl border"
      style={{ borderColor: `${primaryColor}30`, backgroundColor: `${primaryColor}08` }}
    >
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
      >
        <Cpu className="h-4 w-4" style={{ color: primaryColor }} />
      </motion.div>
      <span className="text-sm font-medium" style={{ color: primaryColor }}>
        Pipeline running…
      </span>
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          className="inline-block w-1.5 h-1.5 rounded-full"
          style={{ backgroundColor: primaryColor }}
          animate={{ scale: [1, 1.5, 1], opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 0.8, repeat: Infinity, delay: i * 0.2 }}
        />
      ))}
    </motion.div>
  );
}

// ─── Success Banner (shown briefly after execution completes) ─────

export function SuccessBanner({
  executionTimeMs,
  nodeCount,
}: {
  executionTimeMs?: number;
  nodeCount?: number;
}) {
  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -12 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -12 }}
        transition={{ type: "spring", damping: 20, stiffness: 300 }}
        className="flex items-center gap-3 px-4 py-3 bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl"
      >
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", delay: 0.1 }}
        >
          <CheckCircle2 className="h-5 w-5 text-green-500" />
        </motion.div>
        <div>
          <p className="text-sm font-semibold text-green-800">Pipeline executed successfully!</p>
          <p className="text-xs text-green-600">
            {nodeCount != null ? `${nodeCount} node${nodeCount !== 1 ? "s" : ""} completed` : "All nodes completed"}
            {executionTimeMs != null ? ` in ${executionTimeMs < 1000 ? `${executionTimeMs}ms` : `${(executionTimeMs / 1000).toFixed(1)}s`}` : ""}
          </p>
        </div>
        <div className="ml-auto flex items-center gap-1">
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              className="w-1.5 h-1.5 bg-green-400 rounded-full"
              initial={{ scale: 0 }}
              animate={{ scale: [0, 1.3, 1] }}
              transition={{ delay: 0.15 + i * 0.1, duration: 0.3 }}
            />
          ))}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
