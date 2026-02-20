import {
  Play,
  Trash2,
  Save,
  Settings,
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
    <div className="h-16 bg-white/90 backdrop-blur-xl border-b border-slate-200/60 flex items-center justify-between px-6">
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
          <div className="w-10 h-10 rounded-xl bg-slate-900 flex items-center justify-center">
            <Settings className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-slate-800">
              {projectName || "ML Pipeline Playground"}
            </h1>
            <p className="text-xs text-slate-500">Visual Pipeline Builder</p>
          </div>
        </div>
        <div className="h-10 w-px bg-slate-200/60" />
        {isExecuting && onAbort ? (
          <button
            onClick={onAbort}
            className="ml-3 px-5 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-lg flex items-center gap-2 transition-colors font-semibold"
          >
            <StopCircle className="w-4 h-4" />
            Stop Execution
          </button>
        ) : (
          <button
            onClick={onExecute}
            disabled={isExecuting}
            className="ml-3 px-5 py-2.5 bg-slate-900 hover:bg-slate-800 disabled:bg-slate-300 disabled:cursor-not-allowed text-white rounded-lg flex items-center gap-2 transition-colors font-semibold"
          >
            <Play className="w-4 h-4" />
            {isExecuting ? "Executing..." : "Run Pipeline"}
          </button>
        )}

        {/* Progress Indicator */}
        {executionProgress && (
          <div className="flex items-center gap-3 px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-lg">
            <div className="flex-1 min-w-55">
              <div className="text-xs font-medium text-slate-600 mb-1.5">
                {executionProgress.status}
                {executionProgress.current_node &&
                  executionProgress.total_nodes && (
                    <span className="ml-2 text-slate-500">
                      ({executionProgress.current_node}/
                      {executionProgress.total_nodes} nodes)
                    </span>
                  )}
              </div>
              <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-amber-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${executionProgress.percent}%` }}
                />
              </div>
            </div>
            <div className="text-sm font-bold text-slate-800">
              {executionProgress.percent}%
            </div>
          </div>
        )}
      </div>

      <div className="flex items-center gap-1">
        <button
          onClick={onSave}
          className="px-4 py-2 hover:bg-slate-100 text-slate-500 hover:text-slate-800 rounded-lg flex items-center gap-2 transition-colors"
          title="Save Pipeline"
        >
          <Save className="w-4 h-4" />
          <span className="font-medium">Save</span>
        </button>

        {onExport && (
          <button
            onClick={onExport}
            className="px-4 py-2 hover:bg-slate-100 text-slate-500 hover:text-slate-800 rounded-lg flex items-center gap-2 transition-colors"
            title="Export Pipeline to Code"
          >
            <Code className="w-4 h-4" />
            <span className="font-medium">Export</span>
          </button>
        )}

        {onShare && (
          <button
            onClick={onShare}
            disabled={isShareLoading}
            className="px-4 py-2 hover:bg-slate-100 text-slate-500 hover:text-slate-800 disabled:text-slate-300 disabled:cursor-not-allowed rounded-lg flex items-center gap-2 transition-colors"
            title="Share Project"
          >
            {isShareLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Share2 className="w-4 h-4" />
            )}
            <span className="font-medium">
              {isShareLoading ? "Saving..." : "Share"}
            </span>
          </button>
        )}

        <div className="w-px h-8 bg-slate-200/60 mx-1" />

        {onToggleMentor && (
          <button
            onClick={onToggleMentor}
            className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-colors border ${
              isMentorEnabled
                ? "bg-amber-50 text-amber-600 border-amber-200 hover:bg-amber-100"
                : "text-slate-400 border-transparent hover:bg-slate-100 hover:text-slate-600"
            }`}
            title={isMentorEnabled ? "Disable AI Mentor" : "Enable AI Mentor"}
          >
            <Sparkles className="w-4 h-4" />
            <span className="font-medium">Mentor</span>
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

        <button
          onClick={onClear}
          className="px-4 py-2 hover:bg-red-50 text-slate-500 hover:text-red-600 rounded-lg flex items-center gap-2 transition-colors"
          title="Clear Canvas"
        >
          <Trash2 className="w-4 h-4" />
          <span className="font-medium">Clear</span>
        </button>
      </div>
    </div>
  );
};
