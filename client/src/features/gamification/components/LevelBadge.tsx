/**
 * Circular level number display
 */

import { motion } from "framer-motion";
import { useGamificationStore } from "../store/gamificationStore";

export default function LevelBadge({ size = "md" }: { size?: "sm" | "md" | "lg" }) {
  const { level, progressPercent } = useGamificationStore();

  const sizes = {
    sm: { outer: 32, inner: 26, text: "text-xs", stroke: 2 },
    md: { outer: 44, inner: 36, text: "text-sm", stroke: 3 },
    lg: { outer: 60, inner: 50, text: "text-lg", stroke: 4 },
  };

  const s = sizes[size];
  const radius = (s.outer - s.stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (progressPercent / 100) * circumference;

  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: s.outer, height: s.outer }}>
      {/* Background circle */}
      <svg className="absolute inset-0" width={s.outer} height={s.outer}>
        <circle
          cx={s.outer / 2}
          cy={s.outer / 2}
          r={radius}
          fill="none"
          stroke="#E5E7EB"
          strokeWidth={s.stroke}
        />
        <motion.circle
          cx={s.outer / 2}
          cy={s.outer / 2}
          r={radius}
          fill="none"
          stroke="#F59E0B"
          strokeWidth={s.stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1, ease: "easeOut" }}
          transform={`rotate(-90 ${s.outer / 2} ${s.outer / 2})`}
        />
      </svg>
      {/* Inner circle with level number */}
      <div
        className={`rounded-full bg-gradient-to-br from-amber-50 to-amber-100 flex items-center justify-center font-bold text-amber-700 ${s.text}`}
        style={{ width: s.inner, height: s.inner }}
      >
        {level}
      </div>
    </div>
  );
}
