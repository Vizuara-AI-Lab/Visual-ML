/**
 * Mentor Assistant Component
 *
 * Main floating AI mentor panel that appears in the playground
 */

import React, { useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  Minimize2,
  Maximize2,
  Play,
  Pause,
  MessageCircle,
  HelpCircle,
  Loader2,
  Sparkles,
} from "lucide-react";
import { useMentorStore } from "../store/mentorStore";
import { CharacterAvatar } from "./CharacterAvatar";
import { SuggestionCard } from "./SuggestionCard";
import { GuideCard } from "./GuideCard";
import { useAudioPlayer } from "../hooks/useAudioPlayer";
import { mentorApi } from "../api/mentorApi";
import { toast } from "react-hot-toast";

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
    currentGuide,
    currentSuggestion,
    suggestions,
    setOpen,
    setMinimized,
    showSuggestion,
    dismissSuggestion,
    advanceGuideStep,
  } = useMentorStore();

  const { playText, pause, resume, stop, isSpeaking } = useAudioPlayer();
  const [isLoading, setIsLoading] = React.useState(false);

  // Initial greeting on mount
  useEffect(() => {
    if (preferences.enabled && isOpen) {
      handleGreeting();
    }
  }, []);

  const handleGreeting = async () => {
    try {
      setIsLoading(true);
      const timeOfDay = getTimeOfDay();
      const response = await mentorApi.greetUser(userName, timeOfDay);

      if (response.success && response.suggestions.length > 0) {
        const greetingSuggestion = response.suggestions[0];
        console.log(
          "[Mentor] Showing greeting suggestion:",
          greetingSuggestion,
        );
        showSuggestion(greetingSuggestion);

        // Auto-play greeting if voice-first mode
        console.log("[Mentor] Voice mode:", preferences.voice_mode);
        if (preferences.voice_mode === "voice_first") {
          console.log("[Mentor] Auto-playing greeting audio");
          playText(greetingSuggestion.message);
        } else {
          console.log("[Mentor] Voice-first disabled, not auto-playing");
        }
      }
    } catch (error) {
      console.error("Error getting greeting:", error);
      toast.error("Failed to load mentor greeting");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestionAction = (action: any) => {
    console.log("[MentorAssistant] handleSuggestionAction called:", action);
    onAction?.(action);
  };

  const handlePlayAudio = async (text: string) => {
    console.log(
      "[MentorAssistant] handlePlayAudio called, isSpeaking:",
      isSpeaking,
    );
    if (isSpeaking) {
      console.log("[MentorAssistant] Pausing audio");
      pause();
    } else {
      console.log("[MentorAssistant] Playing audio");
      await playText(text);
    }
  };

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
        className="bg-white/95 backdrop-blur-xl rounded-2xl shadow-2xl border-2 border-indigo-200/60 
                      overflow-hidden flex flex-col"
        style={{ height: isMinimized ? "80px" : "auto" }}
      >
        {/* Header */}
        <div
          className="bg-linear-to-r from-indigo-600 via-purple-600 to-pink-600 px-4 py-3 
                        flex items-center justify-between"
        >
          <div className="flex items-center gap-3">
            <CharacterAvatar size="small" showVoiceIndicator={false} />
            {!isMinimized && (
              <div>
                <h3 className="text-white font-bold text-sm">AI Mentor</h3>
                <p className="text-white/80 text-xs capitalize">
                  {preferences.personality} Mode
                </p>
              </div>
            )}
          </div>

          <div className="flex items-center gap-2">
            {/* Audio Control */}
            {!isMinimized && currentSuggestion && (
              <button
                onClick={() => handlePlayAudio(currentSuggestion.message)}
                className="text-white/80 hover:text-white transition-colors p-1.5 rounded hover:bg-white/10"
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
              className="text-white/80 hover:text-white transition-colors p-1.5 rounded hover:bg-white/10"
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
              className="text-white/80 hover:text-white transition-colors p-1.5 rounded hover:bg-white/10"
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
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-6 h-6 text-indigo-600 animate-spin" />
              </div>
            ) : (
              <>
                {/* Current Guide */}
                <AnimatePresence mode="wait">
                  {currentGuide && (
                    <GuideCard
                      key={currentGuide.modelType}
                      modelType={currentGuide.modelType}
                      steps={currentGuide.steps}
                      explanation={currentGuide.explanation}
                      estimatedTime={currentGuide.estimatedTime}
                      currentStep={currentGuide.currentStep}
                      onStepClick={(nodeType) => {
                        console.log("[Guide] Step clicked:", nodeType);
                        onAction?.({
                          type: "add_node",
                          label: nodeType,
                          payload: { node_type: nodeType },
                        });
                        advanceGuideStep();
                      }}
                    />
                  )}
                </AnimatePresence>

                {/* Current Suggestion */}
                <AnimatePresence mode="wait">
                  {currentSuggestion && !currentGuide && (
                    <SuggestionCard
                      key={currentSuggestion.id}
                      suggestion={currentSuggestion}
                      onAction={handleSuggestionAction}
                      onPlayAudio={() =>
                        handlePlayAudio(currentSuggestion.message)
                      }
                    />
                  )}
                </AnimatePresence>

                {/* Additional Suggestions */}
                {suggestions.slice(1, 3).map((suggestion) => (
                  <SuggestionCard
                    key={suggestion.id}
                    suggestion={suggestion}
                    onAction={handleSuggestionAction}
                    onPlayAudio={() => handlePlayAudio(suggestion.message)}
                  />
                ))}

                {/* Empty State */}
                {suggestions.length === 0 && !isLoading && (
                  <div className="text-center py-8">
                    <Sparkles className="w-12 h-12 text-indigo-300 mx-auto mb-3" />
                    <p className="text-slate-500 text-sm">
                      I'm here to help! Add nodes to your pipeline and I'll
                      provide guidance.
                    </p>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* Footer - Quick Actions */}
        {!isMinimized && (
          <div className="border-t border-slate-200 p-3 bg-slate-50/50">
            <div className="flex gap-2">
              <button
                onClick={handleGreeting}
                className="flex-1 px-3 py-2 bg-white border border-slate-200 rounded-lg 
                           text-xs font-medium text-slate-700 hover:bg-slate-50 
                           transition-colors flex items-center justify-center gap-2"
              >
                <MessageCircle className="w-3.5 h-3.5" />
                Re-greet
              </button>
              <button
                className="flex-1 px-3 py-2 bg-white border border-slate-200 rounded-lg 
                           text-xs font-medium text-slate-700 hover:bg-slate-50 
                           transition-colors flex items-center justify-center gap-2"
                title="Get help"
              >
                <HelpCircle className="w-3.5 h-3.5" />
                Help
              </button>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
};

// Helper function
function getTimeOfDay(): string {
  const hour = new Date().getHours();
  if (hour < 12) return "morning";
  if (hour < 18) return "afternoon";
  return "evening";
}
