/**
 * Guide Card Component
 *
 * Displays step-by-step guide for building ML models
 */

import React from "react";
import { motion } from "framer-motion";
import { Check, Circle, Sparkles, ArrowRight } from "lucide-react";

interface GuideStep {
  step: number;
  node_type: string;
  description: string;
  completed?: boolean;
}

interface GuideCardProps {
  modelType: string;
  steps: GuideStep[];
  explanation: string;
  estimatedTime?: string;
  onStepClick?: (nodeType: string) => void;
  currentStep?: number;
}

export const GuideCard: React.FC<GuideCardProps> = ({
  modelType,
  steps,
  explanation,
  estimatedTime,
  onStepClick,
  currentStep = 0,
}) => {
  const modelTitles: Record<string, string> = {
    linear_regression: "Linear Regression Guide",
    logistic_regression: "Logistic Regression Guide",
    decision_tree: "Decision Tree Guide",
    random_forest: "Random Forest Guide",
  };

  const title = modelTitles[modelType] || "ML Model Guide";

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -20, scale: 0.95 }}
      transition={{ duration: 0.3 }}
      className="relative p-4 rounded-xl border border-amber-200 bg-amber-50"
    >
      {/* Header */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-bold text-slate-900 text-base flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-amber-500" />
            {title}
          </h3>
          {estimatedTime && (
            <span className="text-xs text-amber-700 bg-amber-100 px-2 py-1 rounded-full border border-amber-200">
              {estimatedTime}
            </span>
          )}
        </div>
        <p className="text-sm text-slate-700">{explanation}</p>
      </div>

      {/* Steps */}
      <div className="space-y-2">
        {steps.map((step, index) => {
          const isCompleted = step.completed || index < currentStep;
          const isCurrent = index === currentStep;
          const isUpcoming = index > currentStep;

          return (
            <motion.button
              key={step.step}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              onClick={() => onStepClick?.(step.node_type)}
              className={`w-full flex items-start gap-3 p-3 rounded-lg text-left transition-all
                ${
                  isCompleted
                    ? "bg-green-50 border border-green-200 hover:bg-green-100"
                    : isCurrent
                      ? "bg-white border-2 border-amber-400 hover:bg-amber-50 shadow-sm"
                      : "bg-white border border-slate-200 hover:bg-slate-50"
                }
              `}
            >
              {/* Step indicator */}
              <div
                className={`shrink-0 mt-0.5 ${
                  isCompleted
                    ? "text-green-600"
                    : isCurrent
                      ? "text-amber-600"
                      : "text-slate-400"
                }`}
              >
                {isCompleted ? (
                  <div className="w-6 h-6 rounded-full bg-green-500 flex items-center justify-center">
                    <Check className="w-4 h-4 text-white" />
                  </div>
                ) : (
                  <div
                    className={`w-6 h-6 rounded-full flex items-center justify-center ${
                      isCurrent ? "bg-amber-500" : "bg-slate-200"
                    }`}
                  >
                    <span
                      className={`text-xs font-bold ${
                        isCurrent ? "text-white" : "text-slate-600"
                      }`}
                    >
                      {step.step}
                    </span>
                  </div>
                )}
              </div>

              {/* Step content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p
                      className={`text-sm font-semibold ${
                        isCompleted
                          ? "text-green-900"
                          : isCurrent
                            ? "text-slate-900"
                            : "text-slate-700"
                      }`}
                    >
                      {step.description}
                    </p>
                    <p
                      className={`text-xs mt-0.5 ${
                        isCompleted
                          ? "text-green-600"
                          : isCurrent
                            ? "text-amber-600"
                            : "text-slate-500"
                      }`}
                    >
                      {step.node_type.replace(/_/g, " ")}
                    </p>
                  </div>
                  {isCurrent && (
                    <ArrowRight className="w-4 h-4 text-amber-500 shrink-0 animate-pulse" />
                  )}
                </div>
              </div>
            </motion.button>
          );
        })}
      </div>

      {/* Footer hint */}
      <div className="mt-4 text-xs text-slate-600 bg-white/60 rounded-lg p-2 text-center">
        Click on any step to add that node to your canvas
      </div>
    </motion.div>
  );
};
