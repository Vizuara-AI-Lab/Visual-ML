import {
  Scissors,
  CheckCircle2,
  Target,
  Shuffle,
  Hash,
} from "lucide-react";

interface SplitConfigPanelProps {
  config: Record<string, unknown>;
  onFieldChange: (field: string, value: unknown) => void;
  connectedSourceNode?: { data: { label: string } } | null;
  availableColumns: string[];
}

export function SplitConfigPanel({
  config,
  onFieldChange,
  connectedSourceNode,
  availableColumns,
}: SplitConfigPanelProps) {
  const trainRatio = (config.train_ratio as number) || 0.8;
  const testRatio = 1 - trainRatio;
  const splitType = (config.split_type as string) || "random";
  const shuffle = config.shuffle !== false;
  const targetColumn = (config.target_column as string) || "";

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="bg-linear-to-br from-pink-50 to-rose-50 rounded-xl p-4 border border-pink-100">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-pink-500 flex items-center justify-center shadow-sm">
            <Scissors className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-pink-900">Target & Split</h3>
            <p className="text-xs text-pink-600">
              Select target column and split data into train/test sets
            </p>
          </div>
        </div>
      </div>

      {/* Connection Status */}
      <div
        className={`rounded-xl p-3.5 border ${connectedSourceNode ? "bg-green-50 border-green-200" : "bg-amber-50 border-amber-200"}`}
      >
        <div className="flex items-center gap-2.5">
          {connectedSourceNode ? (
            <CheckCircle2 className="w-4.5 h-4.5 text-green-600" />
          ) : (
            <div className="w-4.5 h-4.5 rounded-full border-2 border-amber-400" />
          )}
          <span
            className={`text-xs font-bold uppercase tracking-wider ${connectedSourceNode ? "text-green-800" : "text-amber-800"}`}
          >
            {connectedSourceNode ? "Data Source Connected" : "No Data Source"}
          </span>
        </div>
        <p
          className={`text-xs mt-1 ${connectedSourceNode ? "text-green-700" : "text-amber-700"}`}
        >
          {connectedSourceNode
            ? `Connected to: ${connectedSourceNode.data.label}`
            : "Connect a data source or preprocessing node"}
        </p>
        {config.dataset_id && (
          <div className="mt-1.5 text-[10px] text-slate-400 font-mono truncate">
            Dataset ID: {config.dataset_id as string}
          </div>
        )}
      </div>

      {/* Target Column */}
      <div className="rounded-xl border border-slate-200 bg-white p-4">
        <div className="flex items-center gap-2 mb-3">
          <Target className="w-4 h-4 text-pink-600" />
          <p className="text-sm font-medium text-slate-700">
            Target Column (y)
          </p>
          <span className="text-red-400 text-xs">*</span>
        </div>
        <select
          value={targetColumn}
          onChange={(e) => onFieldChange("target_column", e.target.value)}
          className="w-full px-3 py-2.5 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-pink-500 focus:border-pink-500"
        >
          <option value="">-- Select Target Column --</option>
          {availableColumns.map((col) => (
            <option key={col} value={col}>
              {col}
            </option>
          ))}
        </select>
        {availableColumns.length === 0 && connectedSourceNode && (
          <p className="text-xs text-amber-600 mt-1.5">
            Run the pipeline first to load available columns
          </p>
        )}
        {!connectedSourceNode && (
          <p className="text-xs text-amber-600 mt-1.5">
            Connect a data source to see available columns
          </p>
        )}
        <p className="text-xs text-slate-400 mt-1.5">
          The column your model will learn to predict
        </p>
      </div>

      {/* Train/Test Split Ratio */}
      <div className="rounded-xl border border-slate-200 bg-white p-4">
        <p className="text-sm font-medium text-slate-700 mb-3">
          Train / Test Split
        </p>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-sm bg-blue-500" />
            <span className="text-sm font-medium text-blue-700">
              Train: {(trainRatio * 100).toFixed(0)}%
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-sm bg-orange-500" />
            <span className="text-sm font-medium text-orange-700">
              Test: {(testRatio * 100).toFixed(0)}%
            </span>
          </div>
        </div>
        {/* Visual bar */}
        <div className="h-3 rounded-full overflow-hidden flex mb-3">
          <div
            className="bg-blue-500 transition-all duration-300"
            style={{ width: `${trainRatio * 100}%` }}
          />
          <div
            className="bg-orange-500 transition-all duration-300"
            style={{ width: `${testRatio * 100}%` }}
          />
        </div>
        <input
          type="range"
          value={trainRatio}
          onChange={(e) =>
            onFieldChange("train_ratio", parseFloat(e.target.value))
          }
          min={0.1}
          max={0.9}
          step={0.05}
          className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"
          style={{
            background: `linear-gradient(to right, #3B82F6 0%, #3B82F6 ${trainRatio * 100}%, #E2E8F0 ${trainRatio * 100}%, #E2E8F0 100%)`,
          }}
        />
        <div className="flex justify-between text-[10px] text-slate-400 mt-1">
          <span>10%</span>
          <span>50%</span>
          <span>90%</span>
        </div>
      </div>

      {/* Split Type */}
      <div className="rounded-xl border border-slate-200 bg-white p-4">
        <p className="text-sm font-medium text-slate-700 mb-3">Split Type</p>
        <div className="grid grid-cols-2 gap-2">
          {[
            {
              value: "random",
              label: "Random",
              icon: Shuffle,
              desc: "Randomly split data",
            },
            {
              value: "stratified",
              label: "Stratified",
              icon: Target,
              desc: "Preserve class ratios",
            },
          ].map((opt) => (
            <button
              key={opt.value}
              onClick={() => onFieldChange("split_type", opt.value)}
              className={`flex flex-col items-center gap-1.5 p-3 rounded-lg border-2 transition-all duration-150 ${
                splitType === opt.value
                  ? "border-pink-500 bg-pink-50 shadow-sm"
                  : "border-slate-200 bg-slate-50 hover:border-slate-300"
              }`}
            >
              <opt.icon
                className={`w-5 h-5 ${splitType === opt.value ? "text-pink-600" : "text-slate-400"}`}
              />
              <span
                className={`text-xs font-medium ${splitType === opt.value ? "text-pink-700" : "text-slate-500"}`}
              >
                {opt.label}
              </span>
              <span className="text-[10px] text-slate-400">{opt.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Random Seed & Shuffle */}
      <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-4">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Hash className="w-4 h-4 text-slate-500" />
            <label className="text-sm font-medium text-slate-700">
              Random Seed
            </label>
          </div>
          <input
            type="number"
            value={(config.random_seed as number) ?? 42}
            onChange={(e) =>
              onFieldChange("random_seed", parseInt(e.target.value) || 42)
            }
            className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-pink-500 focus:border-pink-500"
          />
          <p className="text-xs text-slate-400 mt-1">
            Use the same seed for reproducible splits
          </p>
        </div>

        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <Shuffle className="w-4 h-4 text-slate-500" />
              <span className="text-sm font-medium text-slate-700">
                Shuffle Data
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Randomly shuffle data before splitting
            </p>
          </div>
          <button
            onClick={() => onFieldChange("shuffle", !shuffle)}
            className={`relative w-11 h-6 rounded-full transition-colors duration-200 ${shuffle ? "bg-pink-500" : "bg-slate-300"}`}
          >
            <div
              className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-sm transition-transform duration-200 ${shuffle ? "translate-x-5" : "translate-x-0"}`}
            />
          </button>
        </div>
      </div>

      {/* Ready Status */}
      <div
        className={`rounded-xl p-3.5 border ${targetColumn && connectedSourceNode ? "bg-green-50 border-green-200" : "bg-slate-50 border-slate-200"}`}
      >
        <div className="flex items-center gap-2">
          <CheckCircle2
            className={`w-4 h-4 ${targetColumn && connectedSourceNode ? "text-green-600" : "text-slate-400"}`}
          />
          <span
            className={`text-xs font-semibold ${targetColumn && connectedSourceNode ? "text-green-800" : "text-slate-500"}`}
          >
            {targetColumn && connectedSourceNode
              ? "Ready to split"
              : "Select a target column and connect a data source"}
          </span>
        </div>
      </div>
    </div>
  );
}
