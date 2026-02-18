import { Brain, CheckCircle2, Settings } from "lucide-react";

interface MLAlgorithmConfigPanelProps {
  nodeType: string;
  config: Record<string, unknown>;
  onFieldChange: (field: string, value: unknown) => void;
  connectedSourceNode?: { data: { label: string } } | null;
  availableColumns: string[];
}

const NODE_THEMES: Record<
  string,
  {
    label: string;
    description: string;
    color: string;
    bgFrom: string;
    bgTo: string;
    borderColor: string;
    accentBg: string;
    accentText: string;
    accentTextDark: string;
    accentTextLight: string;
    accentRing: string;
    accentToggle: string;
    accentBorder: string;
    accentSelectedBg: string;
  }
> = {
  linear_regression: {
    label: "Linear Regression",
    description: "Train model for continuous value predictions",
    color: "#3B82F6",
    bgFrom: "from-blue-50",
    bgTo: "to-sky-50",
    borderColor: "border-blue-100",
    accentBg: "bg-blue-500",
    accentText: "text-blue-700",
    accentTextDark: "text-blue-900",
    accentTextLight: "text-blue-600",
    accentRing: "focus:ring-blue-500 focus:border-blue-500",
    accentToggle: "bg-blue-500",
    accentBorder: "border-blue-500",
    accentSelectedBg: "bg-blue-50",
  },
  logistic_regression: {
    label: "Logistic Regression",
    description: "Binary or multi-class classification model",
    color: "#8B5CF6",
    bgFrom: "from-violet-50",
    bgTo: "to-purple-50",
    borderColor: "border-violet-100",
    accentBg: "bg-violet-500",
    accentText: "text-violet-700",
    accentTextDark: "text-violet-900",
    accentTextLight: "text-violet-600",
    accentRing: "focus:ring-violet-500 focus:border-violet-500",
    accentToggle: "bg-violet-500",
    accentBorder: "border-violet-500",
    accentSelectedBg: "bg-violet-50",
  },
  decision_tree: {
    label: "Decision Tree",
    description: "Tree-based classification or regression model",
    color: "#10B981",
    bgFrom: "from-emerald-50",
    bgTo: "to-green-50",
    borderColor: "border-emerald-100",
    accentBg: "bg-emerald-500",
    accentText: "text-emerald-700",
    accentTextDark: "text-emerald-900",
    accentTextLight: "text-emerald-600",
    accentRing: "focus:ring-emerald-500 focus:border-emerald-500",
    accentToggle: "bg-emerald-500",
    accentBorder: "border-emerald-500",
    accentSelectedBg: "bg-emerald-50",
  },
  random_forest: {
    label: "Random Forest",
    description: "Ensemble of decision trees for robust predictions",
    color: "#F59E0B",
    bgFrom: "from-amber-50",
    bgTo: "to-yellow-50",
    borderColor: "border-amber-100",
    accentBg: "bg-amber-500",
    accentText: "text-amber-700",
    accentTextDark: "text-amber-900",
    accentTextLight: "text-amber-600",
    accentRing: "focus:ring-amber-500 focus:border-amber-500",
    accentToggle: "bg-amber-500",
    accentBorder: "border-amber-500",
    accentSelectedBg: "bg-amber-50",
  },
};

export function MLAlgorithmConfigPanel({
  nodeType,
  config,
  onFieldChange,
  connectedSourceNode,
  availableColumns,
}: MLAlgorithmConfigPanelProps) {
  const theme = NODE_THEMES[nodeType] || NODE_THEMES.linear_regression;
  const trainDatasetId = config.train_dataset_id as string;
  const fitIntercept = config.fit_intercept !== false;
  const showAdvanced = (config.show_advanced_options as boolean) || false;

  const hasTreeParams =
    nodeType === "decision_tree" || nodeType === "random_forest";
  const hasFitIntercept =
    nodeType === "linear_regression" || nodeType === "logistic_regression";

  return (
    <div className="space-y-5">
      {/* Header */}
      <div
        className={`bg-linear-to-br ${theme.bgFrom} ${theme.bgTo} rounded-xl p-4 border ${theme.borderColor}`}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center shadow-sm"
            style={{ backgroundColor: theme.color }}
          >
            <Brain className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className={`text-sm font-bold ${theme.accentTextDark}`}>
              {theme.label}
            </h3>
            <p className={`text-xs ${theme.accentTextLight}`}>
              {theme.description}
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
            {connectedSourceNode ? "Split Node Connected" : "No Split Node"}
          </span>
        </div>
        <p
          className={`text-xs mt-1 ${connectedSourceNode ? "text-green-700" : "text-amber-700"}`}
        >
          {connectedSourceNode
            ? `Connected to: ${connectedSourceNode.data.label}`
            : "Connect a Target & Split node"}
        </p>
        {trainDatasetId && (
          <div className="mt-1.5 text-[10px] text-slate-400 font-mono truncate">
            Training Dataset: {trainDatasetId}
          </div>
        )}
        {connectedSourceNode && !trainDatasetId && (
          <p className="text-xs text-amber-600 mt-1">
            Run the split node first to generate training data
          </p>
        )}
      </div>

      {/* Target Column (decision_tree, random_forest) */}
      {hasTreeParams && (
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-sm font-medium text-slate-700">
              Target Column
            </span>
            <span className="text-red-400 text-xs">*</span>
          </div>
          <select
            value={(config.target_column as string) || ""}
            onChange={(e) => onFieldChange("target_column", e.target.value)}
            className={`w-full px-3 py-2.5 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 ${theme.accentRing}`}
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
              Run the pipeline first to load columns
            </p>
          )}
        </div>
      )}

      {/* Task Type (decision_tree, random_forest) */}
      {hasTreeParams && (
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <p className="text-sm font-medium text-slate-700 mb-3">Task Type</p>
          <div className="grid grid-cols-2 gap-2">
            {[
              {
                value: "classification",
                label: "Classification",
                desc: "Predict categories",
              },
              {
                value: "regression",
                label: "Regression",
                desc: "Predict continuous values",
              },
            ].map((opt) => (
              <button
                key={opt.value}
                onClick={() => onFieldChange("task_type", opt.value)}
                className={`flex flex-col items-center gap-1 p-3 rounded-lg border-2 transition-all duration-150 ${
                  (config.task_type || "classification") === opt.value
                    ? `${theme.accentBorder} ${theme.accentSelectedBg} shadow-sm`
                    : "border-slate-200 bg-slate-50 hover:border-slate-300"
                }`}
              >
                <span
                  className={`text-sm font-medium ${(config.task_type || "classification") === opt.value ? theme.accentText : "text-slate-500"}`}
                >
                  {opt.label}
                </span>
                <span className="text-[10px] text-slate-400">{opt.desc}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Fit Intercept (linear_regression, logistic_regression) */}
      {hasFitIntercept && (
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm font-medium text-slate-700">
                Fit Intercept
              </span>
              <p className="text-xs text-slate-400 mt-0.5">
                Calculate the y-intercept for the model
              </p>
            </div>
            <button
              onClick={() => onFieldChange("fit_intercept", !fitIntercept)}
              className={`relative w-11 h-6 rounded-full transition-colors duration-200 ${fitIntercept ? theme.accentToggle : "bg-slate-300"}`}
            >
              <div
                className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-sm transition-transform duration-200 ${fitIntercept ? "translate-x-5" : "translate-x-0"}`}
              />
            </button>
          </div>
        </div>
      )}

      {/* Number of Trees (random_forest only) */}
      {nodeType === "random_forest" && (
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-700">
              Number of Trees
            </span>
            <span className={`text-sm font-bold ${theme.accentText}`}>
              {(config.n_estimators as number) || 100}
            </span>
          </div>
          <input
            type="range"
            value={(config.n_estimators as number) || 100}
            onChange={(e) =>
              onFieldChange("n_estimators", parseInt(e.target.value))
            }
            min={10}
            max={1000}
            step={10}
            className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"
            style={{
              background: `linear-gradient(to right, ${theme.color} 0%, ${theme.color} ${(((config.n_estimators as number) || 100) - 10) / 990 * 100}%, #E2E8F0 ${(((config.n_estimators as number) || 100) - 10) / 990 * 100}%, #E2E8F0 100%)`,
            }}
          />
          <div className="flex justify-between text-[10px] text-slate-400 mt-1">
            <span>10</span>
            <span>500</span>
            <span>1000</span>
          </div>
          <p className="text-xs text-slate-400 mt-1.5">
            More trees = better accuracy but slower training
          </p>
        </div>
      )}

      {/* Tree Hyperparameters (decision_tree, random_forest) */}
      {hasTreeParams && (
        <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-4">
          <p className="text-sm font-medium text-slate-700">
            Tree Hyperparameters
          </p>

          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-medium text-slate-600">
                Max Depth
              </label>
              <span className={`text-xs font-semibold ${theme.accentText}`}>
                {(config.max_depth as number) || "Unlimited"}
              </span>
            </div>
            <input
              type="number"
              value={(config.max_depth as number) || ""}
              onChange={(e) =>
                onFieldChange(
                  "max_depth",
                  e.target.value ? parseInt(e.target.value) : null,
                )
              }
              placeholder="Unlimited"
              min={1}
              max={50}
              className={`w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 ${theme.accentRing}`}
            />
            <p className="text-[10px] text-slate-400 mt-1">
              Leave empty for unlimited depth
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs font-medium text-slate-600 block mb-1.5">
                Min Samples Split
              </label>
              <input
                type="number"
                value={(config.min_samples_split as number) ?? 2}
                onChange={(e) =>
                  onFieldChange(
                    "min_samples_split",
                    parseInt(e.target.value) || 2,
                  )
                }
                min={2}
                max={100}
                className={`w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 ${theme.accentRing}`}
              />
            </div>
            <div>
              <label className="text-xs font-medium text-slate-600 block mb-1.5">
                Min Samples Leaf
              </label>
              <input
                type="number"
                value={(config.min_samples_leaf as number) ?? 1}
                onChange={(e) =>
                  onFieldChange(
                    "min_samples_leaf",
                    parseInt(e.target.value) || 1,
                  )
                }
                min={1}
                max={50}
                className={`w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 ${theme.accentRing}`}
              />
            </div>
          </div>

          <div>
            <label className="text-xs font-medium text-slate-600 block mb-1.5">
              Random State
            </label>
            <input
              type="number"
              value={(config.random_state as number) ?? 42}
              onChange={(e) =>
                onFieldChange("random_state", parseInt(e.target.value) || 42)
              }
              className={`w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 ${theme.accentRing}`}
            />
            <p className="text-[10px] text-slate-400 mt-1">
              Seed for reproducible results
            </p>
          </div>
        </div>
      )}

      {/* Logistic Regression Advanced Options */}
      {nodeType === "logistic_regression" && (
        <>
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Settings className="w-4 h-4 text-slate-500" />
                <span className="text-sm font-medium text-slate-700">
                  Advanced Options
                </span>
              </div>
              <button
                onClick={() =>
                  onFieldChange("show_advanced_options", !showAdvanced)
                }
                className={`relative w-11 h-6 rounded-full transition-colors duration-200 ${showAdvanced ? theme.accentToggle : "bg-slate-300"}`}
              >
                <div
                  className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-sm transition-transform duration-200 ${showAdvanced ? "translate-x-5" : "translate-x-0"}`}
                />
              </button>
            </div>
          </div>

          {showAdvanced && (
            <div className="rounded-xl border border-violet-200 bg-violet-50/50 p-4 space-y-4">
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="text-xs font-medium text-slate-600">
                    Regularization Strength (C)
                  </label>
                  <span className="text-xs font-semibold text-violet-700">
                    {(config.C as number) ?? 1.0}
                  </span>
                </div>
                <input
                  type="number"
                  value={(config.C as number) ?? 1.0}
                  onChange={(e) =>
                    onFieldChange("C", parseFloat(e.target.value) || 1.0)
                  }
                  min={0.001}
                  max={100}
                  step={0.1}
                  className="w-full px-3 py-2 bg-white border border-violet-200 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500"
                />
                <p className="text-[10px] text-slate-400 mt-1">
                  Smaller = stronger regularization
                </p>
              </div>

              <div>
                <label className="text-xs font-medium text-slate-600 block mb-2">
                  Penalty Type
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { value: "l2", label: "L2 (Ridge)" },
                    { value: "l1", label: "L1 (Lasso)" },
                    { value: "none", label: "None" },
                  ].map((opt) => (
                    <button
                      key={opt.value}
                      onClick={() => onFieldChange("penalty", opt.value)}
                      className={`py-2 px-3 rounded-lg border-2 text-xs font-medium transition-all ${
                        (config.penalty || "l2") === opt.value
                          ? "border-violet-500 bg-violet-50 text-violet-700"
                          : "border-slate-200 bg-white text-slate-500 hover:border-slate-300"
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="text-xs font-medium text-slate-600 block mb-2">
                  Solver
                </label>
                <select
                  value={(config.solver as string) || "lbfgs"}
                  onChange={(e) => onFieldChange("solver", e.target.value)}
                  className="w-full px-3 py-2 bg-white border border-violet-200 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500"
                >
                  <option value="lbfgs">LBFGS (recommended)</option>
                  <option value="liblinear">Liblinear</option>
                  <option value="saga">SAGA</option>
                  <option value="newton-cg">Newton-CG</option>
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs font-medium text-slate-600 block mb-1.5">
                    Max Iterations
                  </label>
                  <input
                    type="number"
                    value={(config.max_iter as number) ?? 1000}
                    onChange={(e) =>
                      onFieldChange(
                        "max_iter",
                        parseInt(e.target.value) || 1000,
                      )
                    }
                    min={100}
                    max={10000}
                    className="w-full px-3 py-2 bg-white border border-violet-200 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-slate-600 block mb-1.5">
                    Random State
                  </label>
                  <input
                    type="number"
                    value={(config.random_state as number) ?? 42}
                    onChange={(e) =>
                      onFieldChange(
                        "random_state",
                        parseInt(e.target.value) || 42,
                      )
                    }
                    className="w-full px-3 py-2 bg-white border border-violet-200 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500"
                  />
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {/* Ready Status */}
      <div
        className={`rounded-xl p-3.5 border ${connectedSourceNode && trainDatasetId ? "bg-green-50 border-green-200" : "bg-slate-50 border-slate-200"}`}
      >
        <div className="flex items-center gap-2">
          <CheckCircle2
            className={`w-4 h-4 ${connectedSourceNode && trainDatasetId ? "text-green-600" : "text-slate-400"}`}
          />
          <span
            className={`text-xs font-semibold ${connectedSourceNode && trainDatasetId ? "text-green-800" : "text-slate-500"}`}
          >
            {connectedSourceNode && trainDatasetId
              ? "Ready to train"
              : "Connect split node and run pipeline first"}
          </span>
        </div>
      </div>
    </div>
  );
}
