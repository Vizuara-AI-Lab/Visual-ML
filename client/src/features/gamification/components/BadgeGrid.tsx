/**
 * Grid of badges — earned are colored, locked are greyed
 */

import { motion } from "framer-motion";
import type { Badge } from "../../../types/api";

interface BadgeGridProps {
  badges: Badge[];
}

export default function BadgeGrid({ badges }: BadgeGridProps) {
  if (!badges.length) {
    return (
      <div className="text-center py-8 text-gray-400 text-sm">
        No badges available yet. Keep learning!
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
      {badges.map((badge, i) => (
        <motion.div
          key={badge.badge_id}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.05 }}
          className={`
            relative flex flex-col items-center gap-2 p-3 rounded-xl border transition-all
            ${
              badge.is_earned
                ? "bg-white border-gray-200 shadow-sm hover:shadow-md"
                : "bg-gray-50 border-gray-100 opacity-50 grayscale"
            }
          `}
        >
          <div
            className="w-10 h-10 rounded-full flex items-center justify-center text-xl"
            style={{
              backgroundColor: badge.is_earned ? `${badge.color}18` : "#F3F4F6",
            }}
          >
            {badge.icon}
          </div>
          <div className="text-center">
            <div className="text-xs font-semibold text-gray-800 leading-tight">
              {badge.name}
            </div>
            <div className="text-[10px] text-gray-500 mt-0.5 leading-snug">
              {badge.description}
            </div>
          </div>
          {badge.is_earned && badge.awarded_at && (
            <div className="text-[9px] text-gray-400">
              {new Date(badge.awarded_at).toLocaleDateString()}
            </div>
          )}
          {!badge.is_earned && (
            <div className="absolute top-1.5 right-1.5 w-4 h-4 rounded-full bg-gray-200 flex items-center justify-center">
              <span className="text-[8px] text-gray-400">🔒</span>
            </div>
          )}
        </motion.div>
      ))}
    </div>
  );
}
