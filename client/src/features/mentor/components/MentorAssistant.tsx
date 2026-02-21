/**
 * Mentor Assistant Component
 *
 * Main floating AI mentor panel that appears in the playground.
 * Shows one message at a time, driven by the learning flow state machine.
 */

import React, { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  Minimize2,
  Maximize2,
  Play,
  Pause,
  RotateCcw,
  Sparkles,
} from "lucide-react";
import { useMentorStore } from "../store/mentorStore";
import { CharacterAvatar } from "./CharacterAvatar";
import { SuggestionCard } from "./SuggestionCard";
import { useAudioPlayer } from "../hooks/useAudioPlayer";
import { welcomeMessage } from "../content/learningMessages";
import { ALGORITHM_CONFIG } from "../content/algorithmConfig";

interface MentorAssistantProps {
  userName: string;
  onAction?: (action: any) => void;
}

export const MentorAssistant: React.FC<MentorAssistantProps> = ({
  userName,
  onAction,
}) => {
  const {
    isOpen,
    isMinimized,
    preferences,
    currentSuggestion,
    learningFlow,
    setOpen,
    setMinimized,
    showSuggestion,
    resetFlow,
  } = useMentorStore();

  const { playText, pause, stop, isSpeaking } = useAudioPlayer();

  // ── Show local welcome message when mentor is enabled ──────────

  const hasGreetedRef = useRef(false);

  useEffect(() => {
    if (!preferences.enabled) {
      hasGreetedRef.current = false;
      return;
    }

    if (
      preferences.enabled &&
      learningFlow.stage === "welcome" &&
      !hasGreetedRef.current
    ) {
      hasGreetedRef.current = true;

      const welcome = welcomeMessage(userName);
      const algorithmActions = Object.entries(ALGORITHM_CONFIG).map(
        ([key, config]) => ({
          label: config.displayName,
          type: "select_algorithm" as const,
          payload: { algorithm: key },
        }),
      );

      showSuggestion({
        id: `welcome-${Date.now()}`,
        type: "greeting",
        priority: "info",
        title: welcome.title,
        message: welcome.displayText,
        voice_text: welcome.voiceText,
        actions: algorithmActions,
        dismissible: false,
      });

      // Play greeting audio
      playText(welcome.voiceText, true);
    }
  }, [preferences.enabled, learningFlow.stage, userName, showSuggestion, playText]);

  // ── Auto-play TTS when a new non-greeting suggestion appears ───

  const prevSuggestionIdRef = useRef<string | null>(null);
  useEffect(() => {
    if (
      currentSuggestion &&
      currentSuggestion.id !== prevSuggestionIdRef.current &&
      currentSuggestion.type !== "greeting"
    ) {
      prevSuggestionIdRef.current = currentSuggestion.id;
      const text = currentSuggestion.voice_text || currentSuggestion.message;
      playText(text, true);
    } else if (currentSuggestion) {
      prevSuggestionIdRef.current = currentSuggestion.id;
    }
  }, [currentSuggestion, playText]);

  // ── Stop audio immediately when mentor is disabled ─────────────

  useEffect(() => {
    if (!preferences.enabled) {
      stop();
    }
  }, [preferences.enabled, stop]);

  // ── Handlers ───────────────────────────────────────────────────

  const handleSuggestionAction = (action: any) => {
    onAction?.(action);
  };

  const handlePlayAudio = async (text: string) => {
    if (isSpeaking) {
      pause();
    } else {
      await playText(text);
    }
  };

  const handleStartOver = () => {
    stop();
    resetFlow();
    hasGreetedRef.current = false;
  };

  // ── Render ─────────────────────────────────────────────────────

  if (!preferences.enabled || !isOpen) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: 100, y: 100 }}
      animate={{
        opacity: 1,
        x: 0,
        y: 0,
        width: isMinimized ? 80 : 400,
        height: isMinimized ? 80 : "auto",
      }}
      exit={{ opacity: 0, x: 100, y: 100 }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
      className="fixed bottom-6 right-6 z-40"
      style={{ maxHeight: isMinimized ? "80px" : "calc(100vh - 150px)" }}
    >
      <div
        className="bg-white/95 backdrop-blur-xl rounded-2xl shadow-xl border border-amber-200/60
                      overflow-hidden flex flex-col"
        style={{ height: isMinimized ? "80px" : "auto" }}
      >
        {/* Header */}
        <div
          className="bg-slate-900 px-4 py-3 flex items-center justify-between"
        >
          <div className="flex items-center gap-3">
            <CharacterAvatar size="small" showVoiceIndicator={false} />
            {!isMinimized && (
              <div>
                <h3 className="text-white font-bold text-sm">AI Mentor</h3>
                <p className="text-amber-300/80 text-xs capitalize">
                  {preferences.personality} Mode
                </p>
              </div>
            )}
          </div>

          <div className="flex items-center gap-1">
            {/* Audio Control */}
            {!isMinimized && currentSuggestion && (
              <button
                onClick={() =>
                  handlePlayAudio(
                    currentSuggestion.voice_text ||
                      currentSuggestion.message,
                  )
                }
                className="text-white/70 hover:text-white transition-colors p-1.5 rounded hover:bg-white/10"
                title={isSpeaking ? "Pause" : "Play audio"}
              >
                {isSpeaking ? (
                  <Pause className="w-4 h-4" />
                ) : (
                  <Play className="w-4 h-4" />
                )}
              </button>
            )}

            {/* Minimize/Maximize */}
            <button
              onClick={() => setMinimized(!isMinimized)}
              className="text-white/70 hover:text-white transition-colors p-1.5 rounded hover:bg-white/10"
              title={isMinimized ? "Maximize" : "Minimize"}
            >
              {isMinimized ? (
                <Maximize2 className="w-4 h-4" />
              ) : (
                <Minimize2 className="w-4 h-4" />
              )}
            </button>

            {/* Close */}
            <button
              onClick={() => setOpen(false)}
              className="text-white/70 hover:text-white transition-colors p-1.5 rounded hover:bg-white/10"
              title="Close"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Content */}
        {!isMinimized && (
          <div
            className="flex-1 overflow-y-auto p-4 space-y-3"
            style={{ maxHeight: "500px" }}
          >
            {/* Current Suggestion */}
            <AnimatePresence mode="wait">
              {currentSuggestion && (
                <SuggestionCard
                  key={currentSuggestion.id}
                  suggestion={currentSuggestion}
                  onAction={handleSuggestionAction}
                  onPlayAudio={() =>
                    handlePlayAudio(
                      currentSuggestion.voice_text ||
                        currentSuggestion.message,
                    )
                  }
                />
              )}
            </AnimatePresence>

            {/* Empty State */}
            {!currentSuggestion && (
              <div className="text-center py-8">
                <Sparkles className="w-12 h-12 text-amber-300 mx-auto mb-3" />
                <p className="text-slate-500 text-sm">
                  I'm here to help! Add nodes to your pipeline and I'll
                  provide guidance.
                </p>
              </div>
            )}
          </div>
        )}

        {/* Footer - Start Over */}
        {!isMinimized && learningFlow.stage !== "welcome" && (
          <div className="border-t border-slate-100 p-3 bg-slate-50/50">
            <button
              onClick={handleStartOver}
              className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg
                         text-xs font-medium text-slate-700 hover:bg-slate-50
                         transition-colors flex items-center justify-center gap-2"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Start Over
            </button>
          </div>
        )}
      </div>
    </motion.div>
  );
};
