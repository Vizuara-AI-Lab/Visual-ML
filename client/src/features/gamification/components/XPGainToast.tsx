/**
 * Small toast showing XP gained after an action
 */

import { useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Zap } from "lucide-react";
import { useGamificationStore } from "../store/gamificationStore";

export default function XPGainToast() {
  const { recentXPGain, clearRecentXP } = useGamificationStore();

  useEffect(() => {
    if (recentXPGain !== null) {
      const timer = setTimeout(clearRecentXP, 3000);
      return () => clearTimeout(timer);
    }
  }, [recentXPGain, clearRecentXP]);

  return (
    <AnimatePresence>
      {recentXPGain !== null && (
        <motion.div
          initial={{ opacity: 0, y: 20, scale: 0.9 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -20, scale: 0.9 }}
          transition={{ type: "spring", damping: 20 }}
          className="fixed bottom-6 right-6 z-[9998] flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r from-amber-500 to-orange-500 text-white rounded-xl shadow-lg"
        >
          <Zap className="w-4 h-4 fill-white" />
          <span className="text-sm font-bold">+{recentXPGain} XP</span>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
