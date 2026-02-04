import {
  Play,
  Trash2,
  Save,
  FolderOpen,
  Download,
  Settings,
  ArrowLeft,
} from "lucide-react";

interface ToolbarProps {
  onExecute: () => void;
  onClear: () => void;
  onSave: () => void;
  onLoad: () => void;
  onExport: () => void;
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
  onLoad,
  onExport,
  isExecuting,
  executionProgress,
  projectName,
  onBack,
}: ToolbarProps) => {
  return (
    <div className="h-14 bg-gray-900 border-b border-gray-700 flex items-center justify-between px-4">
      <div className="flex items-center gap-2">
        {onBack && (
          <button
            onClick={onBack}
            className="px-3 py-2 hover:bg-gray-800 text-gray-300 hover:text-white rounded-lg flex items-center gap-2 transition-colors"
            title="Back to Projects"
          >
            <ArrowLeft className="w-4 h-4" />
          </button>
        )}
        <h1 className="text-xl font-bold text-white mr-4">
          {projectName || "ML Pipeline Playground"}
        </h1>

        <button
          onClick={onExecute}
          disabled={isExecuting}
          className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white rounded-lg flex items-center gap-2 transition-colors"
        >
          <Play className="w-4 h-4" />
          {isExecuting ? "Executing..." : "Execute Pipeline"}
        </button>

        {/* Progress Indicator */}
        {executionProgress && (
          <div className="flex items-center gap-3 px-4 py-2 bg-blue-900/30 border border-blue-700/50 rounded-lg">
            <div className="flex-1 min-w-[200px]">
              <div className="text-xs text-blue-300 mb-1">
                {executionProgress.status}
                {executionProgress.current_node &&
                  executionProgress.total_nodes && (
                    <span className="ml-2">
                      ({executionProgress.current_node}/
                      {executionProgress.total_nodes} nodes)
                    </span>
                  )}
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${executionProgress.percent}%` }}
                />
              </div>
            </div>
            <div className="text-sm font-semibold text-blue-400">
              {executionProgress.percent}%
            </div>
          </div>
        )}

        <div className="h-8 w-px bg-gray-700 mx-2" />

        <button
          onClick={onSave}
          className="px-3 py-2 hover:bg-gray-800 text-gray-300 hover:text-white rounded-lg flex items-center gap-2 transition-colors"
          title="Save Pipeline"
        >
          <Save className="w-4 h-4" />
          Save
        </button>

        <button
          onClick={onLoad}
          className="px-3 py-2 hover:bg-gray-800 text-gray-300 hover:text-white rounded-lg flex items-center gap-2 transition-colors"
          title="Load Pipeline"
        >
          <FolderOpen className="w-4 h-4" />
          Load
        </button>

        <button
          onClick={onExport}
          className="px-3 py-2 hover:bg-gray-800 text-gray-300 hover:text-white rounded-lg flex items-center gap-2 transition-colors"
          title="Export Pipeline"
        >
          <Download className="w-4 h-4" />
          Export
        </button>
      </div>

      <div className="flex items-center gap-2">
        <button
          className="px-3 py-2 hover:bg-gray-800 text-gray-300 hover:text-white rounded-lg flex items-center gap-2 transition-colors"
          title="Settings"
        >
          <Settings className="w-4 h-4" />
        </button>

        <button
          onClick={onClear}
          className="px-3 py-2 hover:bg-red-900/20 text-red-400 hover:text-red-300 rounded-lg flex items-center gap-2 transition-colors"
          title="Clear Canvas"
        >
          <Trash2 className="w-4 h-4" />
          Clear
        </button>
      </div>
    </div>
  );
};
