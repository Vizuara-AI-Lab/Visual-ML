/**
 * Fullscreen level-up celebration overlay
 */

import { useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Star, ArrowUp } from "lucide-react";
import { useGamificationStore } from "../store/gamificationStore";

export default function LevelUpAnimation() {
  const { showLevelUp, newLevel, dismissLevelUp } = useGamificationStore();

  const handleDismiss = useCallback(() => {
    dismissLevelUp();
  }, [dismissLevelUp]);

  useEffect(() => {
    if (showLevelUp) {
      const timer = setTimeout(handleDismiss, 4000);
      return () => clearTimeout(timer);
    }
  }, [showLevelUp, handleDismiss]);

  return (
    <AnimatePresence>
      {showLevelUp && newLevel && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
          className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/40 backdrop-blur-sm"
          onClick={handleDismiss}
        >
          <motion.div
            initial={{ scale: 0.3, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.3, opacity: 0 }}
            transition={{ type: "spring", damping: 12, stiffness: 200 }}
            className="relative flex flex-col items-center gap-4 px-12 py-10 bg-white rounded-3xl shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Confetti particles */}
            {Array.from({ length: 12 }).map((_, i) => (
              <motion.div
                key={i}
                className="absolute w-2 h-2 rounded-full"
                style={{
                  backgroundColor: [
                    "#F59E0B",
                    "#EF4444",
                    "#3B82F6",
                    "#10B981",
                    "#8B5CF6",
                    "#EC4899",
                  ][i % 6],
                }}
                initial={{
                  x: 0,
                  y: 0,
                  opacity: 1,
                }}
                animate={{
                  x: Math.cos((i * 30 * Math.PI) / 180) * 120,
                  y: Math.sin((i * 30 * Math.PI) / 180) * 120,
                  opacity: 0,
                  scale: 0,
                }}
                transition={{
                  duration: 1.5,
                  delay: 0.2,
                  ease: "easeOut",
                }}
              />
            ))}

            <motion.div
              animate={{ rotate: [0, 10, -10, 0] }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              <ArrowUp className="w-8 h-8 text-amber-500" />
            </motion.div>

            <div className="text-sm font-semibold text-amber-600 uppercase tracking-wider">
              Level Up!
            </div>

            <motion.div
              className="w-20 h-20 rounded-full bg-gradient-to-br from-amber-400 to-orange-500 flex items-center justify-center shadow-lg"
              animate={{ scale: [1, 1.1, 1] }}
              transition={{ duration: 0.6, delay: 0.4, repeat: 2 }}
            >
              <span className="text-3xl font-black text-white">{newLevel}</span>
            </motion.div>

            <div className="flex items-center gap-1 text-gray-500 text-sm">
              <Star className="w-4 h-4 text-amber-400 fill-amber-400" />
              <span>Keep going! New badges await.</span>
              <Star className="w-4 h-4 text-amber-400 fill-amber-400" />
            </div>

            <button
              onClick={handleDismiss}
              className="mt-2 px-6 py-2 rounded-lg bg-amber-500 hover:bg-amber-600 text-white text-sm font-semibold transition-colors"
            >
              Awesome!
            </button>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
