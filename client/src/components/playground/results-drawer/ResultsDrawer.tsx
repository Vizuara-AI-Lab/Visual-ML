import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { TrendingUp } from "lucide-react";
import { usePlaygroundStore } from "../../../store/playgroundStore";
import { useDrawerResize } from "./hooks/useDrawerResize";
import { DrawerTabBar, type DrawerTab } from "./DrawerTabBar";
import { SummaryTab } from "./tabs/SummaryTab";
import { NodeResultsTab } from "./tabs/NodeResultsTab";
import { MetricsTab } from "./tabs/MetricsTab";
import { LogsTab } from "./tabs/LogsTab";

interface ResultsDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ResultsDrawer = ({ isOpen, onClose }: ResultsDrawerProps) => {
  const { executionResult, isExecuting, nodes, executionLogs } =
    usePlaygroundStore();
  const { height, handleMouseDown, isResizing } = useDrawerResize();
  const [activeTab, setActiveTab] = useState<DrawerTab>("summary");
  const [isMinimized, setIsMinimized] = useState(false);

  const drawerHeight = isMinimized ? 42 : height;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: drawerHeight, opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          transition={{ type: "spring", damping: 25, stiffness: 300 }}
          className="bg-white/95 backdrop-blur-xl border-t border-slate-200/60 shadow-[0_-4px_20px_rgba(0,0,0,0.08)] flex flex-col overflow-hidden"
          style={{ minHeight: 0 }}
        >
          {/* Resize Handle */}
          {!isMinimized && (
            <div
              onMouseDown={handleMouseDown}
              className={`h-1.5 flex items-center justify-center shrink-0 ${
                isResizing
                  ? "cursor-ns-resize bg-slate-100"
                  : "cursor-ns-resize hover:bg-slate-50"
              }`}
            >
              <div className="w-10 h-1 bg-slate-300 rounded-full" />
            </div>
          )}

          {/* Tab Bar */}
          <DrawerTabBar
            activeTab={activeTab}
            onTabChange={(tab) => {
              setActiveTab(tab);
              if (isMinimized) setIsMinimized(false);
            }}
            onMinimize={() => setIsMinimized(!isMinimized)}
            onClose={onClose}
          />

          {/* Content */}
          {!isMinimized && (
            <div className="flex-1 overflow-y-auto">
              {/* Loading State */}
              {isExecuting && (
                <div className="flex items-center justify-center gap-3 py-8">
                  <div className="w-8 h-8 border-3 border-slate-800 border-t-transparent rounded-full animate-spin" />
                  <p className="text-slate-600 text-sm font-medium">
                    Executing pipeline...
                  </p>
                </div>
              )}

              {/* Empty State */}
              {!isExecuting && !executionResult && (
                <div className="flex flex-col items-center justify-center py-8 text-slate-400">
                  <TrendingUp className="w-8 h-8 mb-2" />
                  <p className="text-sm">No results yet</p>
                  <p className="text-xs mt-1">
                    Execute the pipeline to see results
                  </p>
                </div>
              )}

              {/* Tab Content */}
              {!isExecuting && executionResult && (
                <>
                  {activeTab === "summary" && (
                    <SummaryTab executionResult={executionResult} />
                  )}
                  {activeTab === "nodeResults" && (
                    <NodeResultsTab
                      executionResult={executionResult}
                      nodes={nodes}
                    />
                  )}
                  {activeTab === "metrics" && (
                    <MetricsTab
                      metrics={executionResult.metrics || {}}
                    />
                  )}
                  {activeTab === "logs" && (
                    <LogsTab
                      logs={executionLogs}
                      isExecuting={isExecuting}
                    />
                  )}
                </>
              )}

              {/* Show logs tab even during execution */}
              {isExecuting && activeTab === "logs" && (
                <LogsTab logs={executionLogs} isExecuting={isExecuting} />
              )}
            </div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
};
