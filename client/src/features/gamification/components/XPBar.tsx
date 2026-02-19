/**
 * Animated XP progress bar showing level progress
 */

import { motion } from "framer-motion";
import { useGamificationStore } from "../store/gamificationStore";
import { Zap } from "lucide-react";

export default function XPBar({ compact = false }: { compact?: boolean }) {
  const { xp, level, xpToNextLevel, progressPercent } =
    useGamificationStore();

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1 text-xs font-semibold text-amber-600">
          <Zap className="w-3.5 h-3.5" />
          <span>{xp} XP</span>
        </div>
        <div className="w-20 h-1.5 bg-gray-200 rounded-full overflow-hidden">
          <motion.div
            className="h-full rounded-full bg-gradient-to-r from-amber-400 to-amber-500"
            initial={{ width: 0 }}
            animate={{ width: `${progressPercent}%` }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          />
        </div>
        <span className="text-[10px] text-gray-500">Lv {level}</span>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <Zap className="w-4 h-4 text-amber-500" />
          <span className="text-sm font-semibold text-gray-700">
            Level {level}
          </span>
        </div>
        <span className="text-xs text-gray-500">
          {xpToNextLevel} XP to next level
        </span>
      </div>
      <div className="relative h-3 bg-gray-100 rounded-full overflow-hidden border border-gray-200">
        <motion.div
          className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-amber-400 via-amber-500 to-orange-500"
          initial={{ width: 0 }}
          animate={{ width: `${progressPercent}%` }}
          transition={{ duration: 1, ease: "easeOut" }}
        />
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-[10px] font-bold text-gray-700 drop-shadow-sm">
            {xp} XP
          </span>
        </div>
      </div>
    </div>
  );
}
