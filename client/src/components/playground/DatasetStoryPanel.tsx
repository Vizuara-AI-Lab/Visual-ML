/**
 * Slide-out right panel showing dataset narrative, step-by-step guide, and challenge.
 */

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  BookOpen,
  CheckCircle2,
  Circle,
  Trophy,
  ChevronRight,
  Lightbulb,
  Sparkles,
} from "lucide-react";
import type { DatasetStory } from "../../config/datasetStories";

interface DatasetStoryPanelProps {
  story: DatasetStory | null;
  onClose: () => void;
}

export function DatasetStoryPanel({ story, onClose }: DatasetStoryPanelProps) {
  const [completedSteps, setCompletedSteps] = useState<Set<string>>(new Set());
  const [expandedStep, setExpandedStep] = useState<string | null>(null);

  if (!story) return null;

  const toggleStep = (stepId: string) => {
    setCompletedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(stepId)) {
        next.delete(stepId);
      } else {
        next.add(stepId);
      }
      return next;
    });
  };

  const progress = story.steps.length > 0
    ? Math.round((completedSteps.size / story.steps.length) * 100)
    : 0;

  const allComplete = completedSteps.size === story.steps.length;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ x: "100%" }}
        animate={{ x: 0 }}
        exit={{ x: "100%" }}
        transition={{ type: "spring", damping: 25, stiffness: 300 }}
        className="fixed right-0 top-0 bottom-0 w-[380px] bg-white shadow-2xl z-50 flex flex-col border-l border-gray-200"
      >
        {/* Header */}
        <div
          className="px-5 py-4 border-b border-gray-100"
          style={{ backgroundColor: `${story.color}08` }}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <span className="text-2xl">{story.emoji}</span>
              <div>
                <h2 className="text-sm font-bold text-gray-900">
                  {story.title}
                </h2>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <BookOpen className="w-3 h-3" style={{ color: story.color }} />
                  <span className="text-[11px] font-medium" style={{ color: story.color }}>
                    Data Story
                  </span>
                </div>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Progress bar */}
          <div className="mt-3">
            <div className="flex items-center justify-between text-[11px] mb-1">
              <span className="text-gray-500 font-medium">
                {completedSteps.size}/{story.steps.length} steps
              </span>
              <span className="font-semibold" style={{ color: story.color }}>
                {progress}%
              </span>
            </div>
            <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
              <motion.div
                className="h-full rounded-full"
                style={{ backgroundColor: story.color }}
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.4 }}
              />
            </div>
          </div>
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto">
          {/* Narrative */}
          <div className="px-5 py-4 border-b border-gray-50">
            <div className="flex items-start gap-2.5">
              <Sparkles className="w-4 h-4 mt-0.5 shrink-0" style={{ color: story.color }} />
              <p className="text-[13px] text-gray-600 leading-relaxed">
                {story.narrative}
              </p>
            </div>
          </div>

          {/* Steps */}
          <div className="px-5 py-4 space-y-2">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
              Steps
            </h3>
            {story.steps.map((step, idx) => {
              const isComplete = completedSteps.has(step.id);
              const isExpanded = expandedStep === step.id;

              return (
                <div
                  key={step.id}
                  className={`rounded-xl border transition-all ${
                    isComplete
                      ? "bg-green-50/60 border-green-200/60"
                      : "bg-white border-gray-200 hover:border-gray-300"
                  }`}
                >
                  <button
                    onClick={() =>
                      setExpandedStep(isExpanded ? null : step.id)
                    }
                    className="w-full flex items-center gap-3 px-3.5 py-3 text-left"
                  >
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleStep(step.id);
                      }}
                      className="shrink-0"
                    >
                      {isComplete ? (
                        <CheckCircle2 className="w-5 h-5 text-green-500" />
                      ) : (
                        <Circle className="w-5 h-5 text-gray-300 hover:text-gray-400" />
                      )}
                    </button>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span
                          className="text-[10px] font-bold rounded px-1.5 py-0.5"
                          style={{
                            color: story.color,
                            backgroundColor: `${story.color}15`,
                          }}
                        >
                          {idx + 1}
                        </span>
                        <span
                          className={`text-[13px] font-semibold truncate ${
                            isComplete
                              ? "text-green-700 line-through"
                              : "text-gray-800"
                          }`}
                        >
                          {step.title}
                        </span>
                      </div>
                    </div>
                    <ChevronRight
                      className={`w-4 h-4 text-gray-400 shrink-0 transition-transform ${
                        isExpanded ? "rotate-90" : ""
                      }`}
                    />
                  </button>

                  <AnimatePresence>
                    {isExpanded && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="overflow-hidden"
                      >
                        <div className="px-3.5 pb-3.5 pl-12 space-y-2">
                          <p className="text-[12px] text-gray-600 leading-relaxed">
                            {step.description}
                          </p>
                          {step.hint && (
                            <div className="flex items-start gap-2 p-2.5 bg-amber-50 rounded-lg border border-amber-100">
                              <Lightbulb className="w-3.5 h-3.5 text-amber-500 shrink-0 mt-0.5" />
                              <span className="text-[11px] text-amber-700">
                                {step.hint}
                              </span>
                            </div>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              );
            })}
          </div>

          {/* Challenge card */}
          <div className="px-5 pb-5">
            <div
              className={`rounded-xl border-2 p-4 transition-all ${
                allComplete
                  ? "border-amber-300 bg-amber-50"
                  : "border-dashed border-gray-200 bg-gray-50"
              }`}
            >
              <div className="flex items-center gap-2 mb-2">
                <Trophy
                  className={`w-4 h-4 ${
                    allComplete ? "text-amber-500" : "text-gray-400"
                  }`}
                />
                <span
                  className={`text-xs font-bold uppercase tracking-wider ${
                    allComplete ? "text-amber-600" : "text-gray-400"
                  }`}
                >
                  Challenge
                </span>
              </div>
              <p className="text-[12px] text-gray-700 leading-relaxed mb-3">
                {story.challenge.description}
              </p>
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5 px-2.5 py-1 bg-white rounded-lg border border-gray-200">
                  <span className="text-[10px] text-gray-500">Target:</span>
                  <span className="text-[11px] font-bold" style={{ color: story.color }}>
                    {story.challenge.metric} ≥ {story.challenge.threshold}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
