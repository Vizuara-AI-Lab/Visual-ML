import {
  BarChart3,
  Activity,
  TrendingUp,
  Target,
  CheckCircle2,
  type LucideIcon,
} from "lucide-react";

interface ResultMetricsConfigPanelProps {
  nodeType: string;
  config: Record<string, unknown>;
  onFieldChange: (field: string, value: unknown) => void;
  connectedSourceNode?: { data: { label: string } } | null;
}

const NODE_CONFIGS: Record<
  string,
  {
    label: string;
    description: string;
    color: string;
    bgFrom: string;
    bgTo: string;
    borderColor: string;
    icon: LucideIcon;
    hint: string;
  }
> = {
  r2_score: {
    label: "R\u00B2 Score",
    description: "Coefficient of determination (0-1, higher is better)",
    color: "#10B981",
    bgFrom: "from-emerald-50",
    bgTo: "to-green-50",
    borderColor: "border-emerald-100",
    icon: BarChart3,
    hint: "Measures how well predictions fit actual values",
  },
  mse_score: {
    label: "MSE",
    description: "Mean Squared Error (lower is better)",
    color: "#EF4444",
    bgFrom: "from-red-50",
    bgTo: "to-rose-50",
    borderColor: "border-red-100",
    icon: Activity,
    hint: "Average squared difference between predictions and actual",
  },
  rmse_score: {
    label: "RMSE",
    description: "Root Mean Squared Error (lower is better)",
    color: "#F59E0B",
    bgFrom: "from-amber-50",
    bgTo: "to-yellow-50",
    borderColor: "border-amber-100",
    icon: TrendingUp,
    hint: "Square root of MSE, in same units as target variable",
  },
  mae_score: {
    label: "MAE",
    description: "Mean Absolute Error (lower is better)",
    color: "#F97316",
    bgFrom: "from-orange-50",
    bgTo: "to-amber-50",
    borderColor: "border-orange-100",
    icon: BarChart3,
    hint: "Average absolute difference between predictions and actual",
  },
  confusion_matrix: {
    label: "Confusion Matrix",
    description: "Classification predictions vs actual values",
    color: "#8B5CF6",
    bgFrom: "from-violet-50",
    bgTo: "to-purple-50",
    borderColor: "border-violet-100",
    icon: Target,
    hint: "Shows TP, TN, FP, FN for classification models",
  },
  classification_report: {
    label: "Classification Report",
    description: "Precision, recall, F1-score per class",
    color: "#6366F1",
    bgFrom: "from-indigo-50",
    bgTo: "to-blue-50",
    borderColor: "border-indigo-100",
    icon: BarChart3,
    hint: "Detailed classification metrics per class",
  },
  accuracy_score: {
    label: "Accuracy Score",
    description: "Overall classification accuracy",
    color: "#14B8A6",
    bgFrom: "from-teal-50",
    bgTo: "to-emerald-50",
    borderColor: "border-teal-100",
    icon: Target,
    hint: "Percentage of correct predictions",
  },
  roc_curve: {
    label: "ROC Curve",
    description: "Receiver Operating Characteristic curve",
    color: "#EC4899",
    bgFrom: "from-pink-50",
    bgTo: "to-rose-50",
    borderColor: "border-pink-100",
    icon: TrendingUp,
    hint: "Trade-off between true positive and false positive rates",
  },
  feature_importance: {
    label: "Feature Importance",
    description: "Relative importance of each feature",
    color: "#0EA5E9",
    bgFrom: "from-sky-50",
    bgTo: "to-blue-50",
    borderColor: "border-sky-100",
    icon: BarChart3,
    hint: "Shows which features contribute most to predictions",
  },
  residual_plot: {
    label: "Residual Plot",
    description: "Visualization of prediction residuals",
    color: "#84CC16",
    bgFrom: "from-lime-50",
    bgTo: "to-green-50",
    borderColor: "border-lime-100",
    icon: Activity,
    hint: "Shows patterns in prediction errors",
  },
  prediction_table: {
    label: "Prediction Table",
    description: "Actual vs predicted values comparison",
    color: "#64748B",
    bgFrom: "from-slate-100",
    bgTo: "to-gray-50",
    borderColor: "border-slate-200",
    icon: BarChart3,
    hint: "Side-by-side view of actual and predicted values",
  },
};

export function ResultMetricsConfigPanel({
  nodeType,
  config,
  onFieldChange,
  connectedSourceNode,
}: ResultMetricsConfigPanelProps) {
  const nc = NODE_CONFIGS[nodeType] || {
    label: nodeType,
    description: "Result metric",
    color: "#6366F1",
    bgFrom: "from-indigo-50",
    bgTo: "to-blue-50",
    borderColor: "border-indigo-100",
    icon: BarChart3,
    hint: "",
  };

  const NodeIcon = nc.icon;
  const modelOutputId = config.model_output_id as string;

  const hasPrecision =
    nodeType === "mse_score" ||
    nodeType === "rmse_score" ||
    nodeType === "mae_score";

  return (
    <div className="space-y-5">
      {/* Header */}
      <div
        className={`bg-linear-to-br ${nc.bgFrom} ${nc.bgTo} rounded-xl p-4 border ${nc.borderColor}`}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center shadow-sm"
            style={{ backgroundColor: nc.color }}
          >
            <NodeIcon className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-900">{nc.label}</h3>
            <p className="text-xs text-slate-500">{nc.description}</p>
          </div>
        </div>
        {nc.hint && (
          <div className="mt-3 px-3 py-2 bg-white/60 rounded-lg">
            <p className="text-[11px] text-slate-500">{nc.hint}</p>
          </div>
        )}
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
            {connectedSourceNode ? "ML Model Connected" : "No ML Model"}
          </span>
        </div>
        <p
          className={`text-xs mt-1 ${connectedSourceNode ? "text-green-700" : "text-amber-700"}`}
        >
          {connectedSourceNode
            ? `Connected to: ${connectedSourceNode.data.label}`
            : "Connect an ML algorithm node"}
        </p>
        {modelOutputId && (
          <div className="mt-1.5 text-[10px] text-slate-400 font-mono truncate">
            Model Output: {modelOutputId}
          </div>
        )}
        {connectedSourceNode && !modelOutputId && (
          <p className="text-xs text-amber-600 mt-1">
            Run the ML algorithm first to generate model output
          </p>
        )}
      </div>

      {/* R2 Score: Display Format */}
      {nodeType === "r2_score" && (
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <p className="text-sm font-medium text-slate-700 mb-3">
            Display Format
          </p>
          <div className="grid grid-cols-2 gap-2">
            {[
              { value: "percentage", label: "Percentage", example: "75.5%" },
              { value: "decimal", label: "Decimal", example: "0.755" },
            ].map((opt) => (
              <button
                key={opt.value}
                onClick={() => onFieldChange("display_format", opt.value)}
                className={`flex flex-col items-center gap-1 p-3 rounded-lg border-2 transition-all duration-150 ${
                  (config.display_format || "percentage") === opt.value
                    ? "border-emerald-500 bg-emerald-50 shadow-sm"
                    : "border-slate-200 bg-slate-50 hover:border-slate-300"
                }`}
              >
                <span
                  className={`text-lg font-bold ${(config.display_format || "percentage") === opt.value ? "text-emerald-700" : "text-slate-400"}`}
                >
                  {opt.example}
                </span>
                <span
                  className={`text-xs font-medium ${(config.display_format || "percentage") === opt.value ? "text-emerald-600" : "text-slate-500"}`}
                >
                  {opt.label}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Precision slider for MSE, RMSE, MAE */}
      {hasPrecision && (
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-700">
              Decimal Precision
            </span>
            <span
              className="text-sm font-bold"
              style={{ color: nc.color }}
            >
              {(config.precision as number) ?? 4}
            </span>
          </div>
          <input
            type="range"
            value={(config.precision as number) ?? 4}
            onChange={(e) =>
              onFieldChange("precision", parseInt(e.target.value))
            }
            min={1}
            max={8}
            step={1}
            className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"
            style={{
              background: `linear-gradient(to right, ${nc.color} 0%, ${nc.color} ${(((config.precision as number) ?? 4) - 1) / 7 * 100}%, #E2E8F0 ${(((config.precision as number) ?? 4) - 1) / 7 * 100}%, #E2E8F0 100%)`,
            }}
          />
          <div className="flex justify-between text-[10px] text-slate-400 mt-1">
            <span>1</span>
            <span>4</span>
            <span>8</span>
          </div>
          <p className="text-xs text-slate-400 mt-1.5">
            Number of decimal places to display
          </p>
        </div>
      )}

      {/* Confusion Matrix: Show Percentages & Color Scheme */}
      {nodeType === "confusion_matrix" && (
        <>
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-sm font-medium text-slate-700">
                  Show Percentages
                </span>
                <p className="text-xs text-slate-400 mt-0.5">
                  Display percentages alongside counts
                </p>
              </div>
              <button
                onClick={() =>
                  onFieldChange(
                    "show_percentages",
                    !(config.show_percentages !== false),
                  )
                }
                className={`relative w-11 h-6 rounded-full transition-colors duration-200 ${config.show_percentages !== false ? "bg-violet-500" : "bg-slate-300"}`}
              >
                <div
                  className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-sm transition-transform duration-200 ${config.show_percentages !== false ? "translate-x-5" : "translate-x-0"}`}
                />
              </button>
            </div>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <p className="text-sm font-medium text-slate-700 mb-3">
              Color Scheme
            </p>
            <div className="grid grid-cols-3 gap-2">
              {[
                {
                  value: "blue",
                  label: "Blue",
                  colors: "from-blue-400 to-blue-600",
                },
                {
                  value: "green",
                  label: "Green",
                  colors: "from-green-400 to-green-600",
                },
                {
                  value: "purple",
                  label: "Purple",
                  colors: "from-purple-400 to-purple-600",
                },
              ].map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => onFieldChange("color_scheme", opt.value)}
                  className={`flex flex-col items-center gap-2 p-3 rounded-lg border-2 transition-all duration-150 ${
                    (config.color_scheme || "blue") === opt.value
                      ? "border-violet-500 shadow-sm"
                      : "border-slate-200 hover:border-slate-300"
                  }`}
                >
                  <div
                    className={`w-8 h-8 rounded-md bg-linear-to-br ${opt.colors}`}
                  />
                  <span
                    className={`text-xs font-medium ${(config.color_scheme || "blue") === opt.value ? "text-violet-700" : "text-slate-500"}`}
                  >
                    {opt.label}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </>
      )}

      {/* Ready Status */}
      <div
        className={`rounded-xl p-3.5 border ${connectedSourceNode && modelOutputId ? "bg-green-50 border-green-200" : "bg-slate-50 border-slate-200"}`}
      >
        <div className="flex items-center gap-2">
          <CheckCircle2
            className={`w-4 h-4 ${connectedSourceNode && modelOutputId ? "text-green-600" : "text-slate-400"}`}
          />
          <span
            className={`text-xs font-semibold ${connectedSourceNode && modelOutputId ? "text-green-800" : "text-slate-500"}`}
          >
            {connectedSourceNode && modelOutputId
              ? "Ready to evaluate"
              : "Connect an ML model and run it first"}
          </span>
        </div>
      </div>
    </div>
  );
}
