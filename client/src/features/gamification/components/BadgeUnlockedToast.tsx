/**
 * Toast notification when a badge is unlocked
 */

import { useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Award } from "lucide-react";
import { useGamificationStore } from "../store/gamificationStore";

export default function BadgeUnlockedToast() {
  const { recentBadges, clearRecentBadges } = useGamificationStore();

  useEffect(() => {
    if (recentBadges.length > 0) {
      const timer = setTimeout(clearRecentBadges, 5000);
      return () => clearTimeout(timer);
    }
  }, [recentBadges, clearRecentBadges]);

  return (
    <div className="fixed top-4 right-4 z-[9999] flex flex-col gap-2">
      <AnimatePresence>
        {recentBadges.map((badge) => (
          <motion.div
            key={badge.badge_id}
            initial={{ opacity: 0, x: 100, scale: 0.8 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 100, scale: 0.8 }}
            transition={{ type: "spring", damping: 20 }}
            className="flex items-center gap-3 px-4 py-3 bg-white rounded-xl shadow-lg border border-amber-200 min-w-[260px]"
          >
            <div
              className="w-10 h-10 rounded-full flex items-center justify-center text-lg shrink-0"
              style={{ backgroundColor: `${badge.color}20` }}
            >
              {badge.icon}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1">
                <Award className="w-3.5 h-3.5 text-amber-500" />
                <span className="text-xs font-semibold text-amber-600">
                  Badge Unlocked!
                </span>
              </div>
              <div className="text-sm font-bold text-gray-800 truncate">
                {badge.name}
              </div>
              <div className="text-[11px] text-gray-500 truncate">
                {badge.description}
              </div>
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
