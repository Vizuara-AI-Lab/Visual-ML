/**
 * Suggestion Card Component
 *
 * Displays individual mentor suggestions with actions
 */

import React from "react";
import { motion } from "framer-motion";
import { X, AlertCircle, Info, AlertTriangle, Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import type { MentorSuggestion } from "../store/mentorStore";
import { useMentorStore } from "../store/mentorStore";

interface SuggestionCardProps {
  suggestion: MentorSuggestion;
  onAction?: (action: MentorSuggestion["actions"][0]) => void;
  onPlayAudio?: () => void;
}

export const SuggestionCard: React.FC<SuggestionCardProps> = ({
  suggestion,
  onAction,
  onPlayAudio,
}) => {
  const dismissSuggestion = useMentorStore((state) => state.dismissSuggestion);

  const priorityConfig = {
    info: {
      icon: Info,
      bgColor: "bg-blue-50",
      borderColor: "border-blue-200",
      iconColor: "text-blue-600",
      buttonColor: "bg-blue-600 hover:bg-blue-700",
    },
    warning: {
      icon: AlertTriangle,
      bgColor: "bg-yellow-50",
      borderColor: "border-yellow-200",
      iconColor: "text-yellow-600",
      buttonColor: "bg-yellow-600 hover:bg-yellow-700",
    },
    critical: {
      icon: AlertCircle,
      bgColor: "bg-red-50",
      borderColor: "border-red-200",
      iconColor: "text-red-600",
      buttonColor: "bg-red-600 hover:bg-red-700",
    },
  };

  const config = priorityConfig[suggestion.priority];
  const Icon = config.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -20, scale: 0.95 }}
      transition={{ duration: 0.3 }}
      className={`relative p-4 rounded-xl border-2 ${config.borderColor} ${config.bgColor} shadow-sm`}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <div
            className={`w-8 h-8 rounded-lg ${config.iconColor.replace("text-", "bg-").replace("600", "100")} 
                          flex items-center justify-center`}
          >
            <Icon className={`w-5 h-5 ${config.iconColor}`} />
          </div>
          <h3 className="font-bold text-slate-800 text-sm">
            {suggestion.title}
          </h3>
        </div>

        {suggestion.dismissible && (
          <button
            onClick={() => dismissSuggestion(suggestion.id)}
            className="text-slate-400 hover:text-slate-600 transition-colors"
            aria-label="Dismiss"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Message */}
      <div className="prose prose-sm max-w-none text-slate-700 mb-3">
        <ReactMarkdown>{suggestion.message}</ReactMarkdown>
      </div>

      {/* Actions */}
      {suggestion.actions && suggestion.actions.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {suggestion.actions.map((action, index) => (
            <button
              key={index}
              onClick={() => {
                console.log(
                  "[SuggestionCard] Button clicked:",
                  action.label,
                  action,
                );
                onAction?.(action);
              }}
              className={`px-3 py-1.5 rounded-lg text-white text-xs font-semibold 
                         ${config.buttonColor} transition-colors shadow-sm hover:shadow-md 
                         flex items-center gap-1.5`}
            >
              <Sparkles className="w-3 h-3" />
              {action.label}
            </button>
          ))}
        </div>
      )}

      {/* Timestamp */}
      {suggestion.timestamp && (
        <div className="mt-3 text-xs text-slate-400">
          {new Date(suggestion.timestamp).toLocaleTimeString()}
        </div>
      )}
    </motion.div>
  );
};
