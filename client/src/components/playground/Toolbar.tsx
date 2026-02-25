import {
  Play,
  Trash2,
  Save,
  Workflow,
  ArrowLeft,
  Share2,
  StopCircle,
  Code,
  Sparkles,
  Loader2,
} from "lucide-react";

interface ToolbarProps {
  onExecute: () => void;
  onAbort?: () => void;
  onClear: () => void;
  onSave: () => void;
  onShare?: () => void;
  onExport?: () => void;
  isExecuting: boolean;
  isShareLoading?: boolean;
  executionProgress?: {
    status: string;
    percent: number;
    current_node?: number;
    total_nodes?: number;
  } | null;
  projectName?: string;
  onBack?: () => void;
  isMentorEnabled?: boolean;
  onToggleMentor?: () => void;
}

export const Toolbar = ({
  onExecute,
  onAbort,
  onClear,
  onSave,
  onShare,
  onExport,
  isExecuting,
  isShareLoading,
  executionProgress,
  projectName,
  onBack,
  isMentorEnabled,
  onToggleMentor,
}: ToolbarProps) => {
  return (
    <div className="h-16 bg-white/90 backdrop-blur-xl border-b border-slate-200/60 shadow-sm shadow-slate-900/5 flex items-center justify-between px-6">
      <div className="flex items-center gap-3">
        {onBack && (
          <button
            onClick={onBack}
            className="p-2 hover:bg-slate-100 text-slate-500 hover:text-slate-800 rounded-lg transition-colors"
            title="Back to Projects"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
        )}

        <div className="flex items-center gap-3 mr-6">
          <div className="w-10 h-10 rounded-xl bg-linear-to-br from-slate-800 to-slate-950 flex items-center justify-center shadow-md shadow-slate-900/20">
            <Workflow className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-slate-800">
              {projectName || "ML Pipeline Playground"}
            </h1>
            <p className="text-xs text-slate-400 font-medium tracking-wide">Visual Pipeline Builder</p>
          </div>
        </div>
        <div className="h-10 w-px bg-slate-200/60" />

        {isExecuting && onAbort ? (
          <button
            onClick={onAbort}
            className="ml-3 px-5 py-2.5 bg-linear-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white rounded-lg flex items-center gap-2 transition-all font-semibold shadow-md shadow-red-500/25 hover:shadow-lg hover:shadow-red-500/30"
          >
            <StopCircle className="w-4 h-4" />
            Stop Execution
          </button>
        ) : (
          <button
            onClick={onExecute}
            disabled={isExecuting}
            className="ml-3 px-5 py-2.5 bg-linear-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 disabled:from-slate-300 disabled:to-slate-300 disabled:cursor-not-allowed disabled:shadow-none text-white rounded-lg flex items-center gap-2 transition-all font-semibold shadow-md shadow-emerald-500/25 hover:shadow-lg hover:shadow-emerald-500/30"
          >
            <Play className="w-4 h-4" />
            {isExecuting ? "Executing..." : "Run Pipeline"}
          </button>
        )}

        {/* Progress Indicator */}
        {executionProgress && (
          <div className="flex items-center gap-3 px-4 py-2.5 bg-emerald-50/80 border border-emerald-200/60 rounded-lg">
            <Loader2 className="w-4 h-4 text-emerald-600 animate-spin shrink-0" />
            <div className="flex-1 min-w-55">
              <div className="text-xs font-medium text-emerald-700 mb-1.5">
                {executionProgress.status}
                {executionProgress.current_node &&
                  executionProgress.total_nodes && (
                    <span className="ml-2 text-emerald-500">
                      ({executionProgress.current_node}/
                      {executionProgress.total_nodes} nodes)
                    </span>
                  )}
              </div>
              <div className="w-full bg-emerald-100 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-linear-to-r from-emerald-400 to-emerald-500 h-2 rounded-full transition-all duration-300 animate-pulse"
                  style={{ width: `${executionProgress.percent}%` }}
                />
              </div>
            </div>
            <div className="text-sm font-bold text-emerald-700">
              {executionProgress.percent}%
            </div>
          </div>
        )}
      </div>

      <div className="flex items-center gap-2">
        {/* Action button group */}
        <div className="flex items-center bg-slate-50/80 border border-slate-200/60 rounded-lg p-1 gap-0.5">
          <button
            onClick={onSave}
            className="p-2 hover:bg-white text-slate-500 hover:text-slate-700 rounded-md flex items-center justify-center transition-colors"
            title="Save Pipeline"
          >
            <Save className="w-4 h-4" />
          </button>

          {onExport && (
            <button
              onClick={onExport}
              className="p-2 hover:bg-white text-slate-500 hover:text-slate-700 rounded-md flex items-center justify-center transition-colors"
              title="Export Pipeline to Code"
            >
              <Code className="w-4 h-4" />
            </button>
          )}

          {onShare && (
            <button
              onClick={onShare}
              disabled={isShareLoading}
              className="p-2 hover:bg-white text-slate-500 hover:text-slate-700 disabled:text-slate-300 disabled:cursor-not-allowed rounded-md flex items-center justify-center transition-colors"
              title={isShareLoading ? "Saving..." : "Share Project"}
            >
              {isShareLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Share2 className="w-4 h-4" />
              )}
            </button>
          )}
        </div>

        {/* Mentor toggle */}
        {onToggleMentor && (
          <button
            onClick={onToggleMentor}
            className={`px-3 py-2 rounded-lg flex items-center gap-2 transition-colors border ${
              isMentorEnabled
                ? "bg-amber-50 text-amber-600 border-amber-200 hover:bg-amber-100"
                : "text-slate-400 border-transparent hover:bg-slate-100 hover:text-slate-600"
            }`}
            title={isMentorEnabled ? "Disable AI Mentor" : "Enable AI Mentor"}
          >
            <Sparkles className="w-4 h-4" />
            <span className="text-sm font-medium">Mentor</span>
            <span
              className={`text-[10px] font-bold uppercase px-1.5 py-0.5 rounded-full ${
                isMentorEnabled
                  ? "bg-amber-200/60 text-amber-700"
                  : "bg-slate-200/60 text-slate-500"
              }`}
            >
              {isMentorEnabled ? "ON" : "OFF"}
            </span>
          </button>
        )}

        {/* Clear — de-emphasized icon-only */}
        <button
          onClick={onClear}
          className="p-2 hover:bg-red-50 text-slate-400 hover:text-red-500 rounded-lg flex items-center justify-center transition-colors"
          title="Clear Canvas"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};
