import { Play, Trash2, Save, Settings, ArrowLeft } from "lucide-react";

interface ToolbarProps {
  onExecute: () => void;
  onClear: () => void;
  onSave: () => void;
  isExecuting: boolean;
  executionProgress?: {
    status: string;
    percent: number;
    current_node?: number;
    total_nodes?: number;
  } | null;
  projectName?: string;
  onBack?: () => void;
}

export const Toolbar = ({
  onExecute,
  onClear,
  onSave,
  isExecuting,
  executionProgress,
  projectName,
  onBack,
}: ToolbarProps) => {
  return (
    <div className="h-16 bg-white/90 backdrop-blur-xl border-b border-slate-200/60 flex items-center justify-between px-6 shadow-lg shadow-slate-900/5">
      <div className="flex items-center gap-3">
        {onBack && (
          <button
            onClick={onBack}
            className="p-2 hover:bg-slate-100/50 text-slate-500 hover:text-slate-800 rounded-lg transition-all"
            title="Back to Projects"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
        )}

        <div className="flex items-center gap-3 mr-6">
          <div className="w-10 h-10 rounded-xl bg-slate-900 flex items-center justify-center shadow-lg shadow-slate-900/25">
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
        <button
          onClick={onExecute}
          disabled={isExecuting}
          className="ml-3 px-5 py-2.5 bg-slate-900 hover:bg-slate-800 disabled:bg-slate-300 disabled:cursor-not-allowed text-white rounded-lg flex items-center gap-2 transition-all shadow-lg shadow-slate-900/25 hover:shadow-xl hover:shadow-slate-900/30 disabled:shadow-none font-semibold"
        >
          <Play className="w-4 h-4" />
          {isExecuting ? "Executing..." : "Run Pipeline"}
        </button>

        {/* Progress Indicator */}
        {executionProgress && (
          <div className="flex items-center gap-3 px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-lg shadow-md">
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
                  className="bg-slate-900 h-2 rounded-full transition-all duration-300"
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

      <div className="flex items-center gap-2">
        <button
          onClick={onSave}
          className="px-4 py-2 hover:bg-slate-100/50 text-slate-500 hover:text-slate-800 rounded-lg flex items-center gap-2 transition-all"
          title="Save Pipeline"
        >
          <Save className="w-4 h-4" />
          <span className="font-medium">Save</span>
        </button>

        <button
          onClick={onClear}
          className="px-4 py-2 hover:bg-red-50 text-red-600 hover:text-red-700 rounded-lg flex items-center gap-2 transition-all border border-transparent hover:border-red-200"
          title="Clear Canvas"
        >
          <Trash2 className="w-4 h-4" />
          <span className="font-medium">Clear</span>
        </button>
      </div>
    </div>
  );
};
