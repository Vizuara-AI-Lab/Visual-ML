/**
 * Character Avatar Component
 *
 * Displays the AI mentor character with animated states (idle, speaking)
 */

import { motion } from "framer-motion";
import { Volume2, VolumeX } from "lucide-react";
import { useMentorStore } from "../store/mentorStore";

interface CharacterAvatarProps {
  size?: "small" | "medium" | "large";
  showVoiceIndicator?: boolean;
}

export const CharacterAvatar: React.FC<CharacterAvatarProps> = ({
  size = "medium",
  showVoiceIndicator = true,
}) => {
  const { preferences, isSpeaking } = useMentorStore();

  const sizeClasses = {
    small: "w-12 h-12",
    medium: "w-20 h-20",
    large: "w-32 h-32",
  };

  const avatarImages: Record<string, string> = {
    scientist: "🧑‍🔬",
    teacher: "👨‍🏫",
    robot: "🤖",
    wizard: "🧙‍♂️",
    coach: "👨‍💼",
    engineer: "👷",
  };

  return (
    <div className="relative">
      {/* Avatar Container */}
      <motion.div
        className={`${sizeClasses[size]} rounded-full bg-amber-500
                   flex items-center justify-center shadow-md relative overflow-hidden`}
        animate={{
          scale: isSpeaking ? [1, 1.05, 1] : [1, 1.02, 1],
        }}
        transition={{
          duration: isSpeaking ? 0.5 : 2,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      >
        {/* Glow effect when speaking */}
        {isSpeaking && (
          <motion.div
            className="absolute inset-0 bg-amber-400"
            animate={{
              opacity: [0.3, 0.7, 0.3],
            }}
            transition={{
              duration: 1,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
        )}

        {/* Avatar Emoji */}
        <span
          className={`relative z-10 ${size === "large" ? "text-6xl" : size === "medium" ? "text-4xl" : "text-2xl"}`}
        >
          {avatarImages[preferences.avatar] || avatarImages.scientist}
        </span>

        {/* Pulse ring when speaking */}
        {isSpeaking && (
          <motion.div
            className="absolute inset-0 rounded-full border-4 border-amber-400"
            animate={{
              scale: [1, 1.3],
              opacity: [0.7, 0],
            }}
            transition={{
              duration: 1,
              repeat: Infinity,
              ease: "easeOut",
            }}
          />
        )}
      </motion.div>

      {/* Voice Indicator */}
      {showVoiceIndicator && (
        <motion.div
          className="absolute -bottom-1 -right-1 w-7 h-7 rounded-full bg-white shadow-md
                     flex items-center justify-center"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", stiffness: 300, damping: 20 }}
        >
          {isSpeaking ? (
            <Volume2 className="w-4 h-4 text-amber-600" />
          ) : (
            <VolumeX className="w-4 h-4 text-slate-400" />
          )}
        </motion.div>
      )}
    </div>
  );
};
