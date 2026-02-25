/**
 * MLPClassifierExplorer - Interactive learning activities for MLP Classifier
 * Tabbed component: Results | Network Architecture | Training Progress | Class Distribution | Metrics Explained | Quiz | How It Works
 */

import { useState, useMemo } from "react";
import {
  ClipboardList,
  Network,
  TrendingDown,
  BarChart3,
  Gauge,
  HelpCircle,
  Cog,
  CheckCircle,
  XCircle,
  Trophy,
  Target,
  Timer,
  Hash,
  Database,
  Layers,
  Brain,
  Zap,
  AlertTriangle,
} from "lucide-react";

interface MLPClassifierExplorerProps {
  result: any;
}

type ExplorerTab =
  | "results"
  | "architecture"
  | "loss_curve"
  | "class_distribution"
  | "metric_explainer"
  | "quiz"
  | "how_it_works";

export const MLPClassifierExplorer = ({
  result,
}: MLPClassifierExplorerProps) => {
  const [activeTab, setActiveTab] = useState<ExplorerTab>("results");

  const tabs: {
    id: ExplorerTab;
    label: string;
    icon: any;
    available: boolean;
  }[] = [
    { id: "results", label: "Results", icon: ClipboardList, available: true },
    {
      id: "architecture",
      label: "Network Architecture",
      icon: Network,
      available: !!result.architecture,
    },
    {
      id: "loss_curve",
      label: "Training Progress",
      icon: TrendingDown,
      available: !!result.loss_curve_analysis?.has_data,
    },
    {
      id: "class_distribution",
      label: "Class Distribution",
      icon: BarChart3,
      available: !!result.class_distribution?.classes?.length,
    },
    {
      id: "metric_explainer",
      label: "Metrics Explained",
      icon: Gauge,
      available: !!result.metric_explainer?.metrics?.length,
    },
    {
      id: "quiz",
      label: "Quiz",
      icon: HelpCircle,
      available: !!result.quiz_questions && result.quiz_questions.length > 0,
    },
    {
      id: "how_it_works",
      label: "How It Works",
      icon: Cog,
      available: true,
    },
  ];

  const metrics = result.training_metrics || {};

  return (
    <div className="space-y-4">
      {/* Tab navigation */}
      <div className="flex border-b border-gray-200 overflow-x-auto">
        {tabs
          .filter((t) => t.available)
          .map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                  activeTab === tab.id
                    ? "border-violet-500 text-violet-700"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
      </div>

      {/* Tab content */}
      <div className="min-h-[400px]">
        {activeTab === "results" && (
          <ResultsTab result={result} metrics={metrics} />
        )}
        {activeTab === "architecture" && (
          <ArchitectureTab result={result} />
        )}
        {activeTab === "loss_curve" && (
          <LossCurveTab data={result.loss_curve_analysis} lossCurve={result.loss_curve} validationScores={result.validation_scores} />
        )}
        {activeTab === "class_distribution" && (
          <ClassDistributionTab data={result.class_distribution} />
        )}
        {activeTab === "metric_explainer" && (
          <MetricExplainerTab data={result.metric_explainer} />
        )}
        {activeTab === "quiz" && (
          <QuizTab questions={result.quiz_questions || []} />
        )}
        {activeTab === "how_it_works" && (
          <HowItWorksTab architecture={result.architecture} activation={result.metadata?.activation} />
        )}
      </div>
    </div>
  );
};

// ======================== Results Tab ========================

function ResultsTab({ result, metrics }: { result: any; metrics: any }) {
  const accuracy = metrics.accuracy;
  const accuracyPct = accuracy != null ? accuracy * 100 : null;

  return (
    <div className="space-y-6">
      {/* Status Banner */}
      <div className="bg-gradient-to-r from-violet-50 to-purple-50 border border-violet-200 rounded-lg p-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-violet-500 rounded-full flex items-center justify-center">
            <CheckCircle className="text-white w-5 h-5" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-violet-900">
              MLP Classifier Training Complete
            </h3>
            <p className="text-sm text-violet-700">
              Neural network with {result.architecture?.hidden_layers?.length || "?"} hidden layer(s) trained on{" "}
              {result.training_samples?.toLocaleString()} samples — {result.n_classes} classes detected
            </p>
          </div>
        </div>
      </div>

      {/* Hero Metric */}
      {accuracyPct != null && (
        <div className="flex justify-center">
          <div className="relative w-36 h-36">
            <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
              <circle cx="60" cy="60" r="52" fill="none" stroke="#e5e7eb" strokeWidth="8" />
              <circle
                cx="60"
                cy="60"
                r="52"
                fill="none"
                stroke={accuracyPct >= 80 ? "#8b5cf6" : accuracyPct >= 60 ? "#f59e0b" : "#ef4444"}
                strokeWidth="8"
                strokeLinecap="round"
                strokeDasharray={`${(accuracyPct / 100) * 327} 327`}
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-2xl font-bold text-gray-900">
                {accuracyPct.toFixed(1)}%
              </span>
              <span className="text-xs text-gray-500">Accuracy</span>
            </div>
          </div>
        </div>
      )}

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wide mb-1">
            <Layers className="w-3.5 h-3.5" /> Layers
          </div>
          <div className="text-lg font-semibold text-gray-900">
            {result.architecture?.hidden_layers?.length || "?"}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wide mb-1">
            <Database className="w-3.5 h-3.5" /> Samples
          </div>
          <div className="text-lg font-semibold text-gray-900">
            {result.training_samples?.toLocaleString()}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wide mb-1">
            <Hash className="w-3.5 h-3.5" /> Features
          </div>
          <div className="text-lg font-semibold text-gray-900">{result.n_features}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wide mb-1">
            <Timer className="w-3.5 h-3.5" /> Training Time
          </div>
          <div className="text-lg font-semibold text-gray-900">
            {result.training_time_seconds?.toFixed(2)}s
          </div>
        </div>
      </div>

      {/* All Metrics */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-900 mb-4">Training Metrics</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(metrics)
            .filter(([key, value]) => typeof value === "number" && key !== "confusion_matrix")
            .map(([key, value]) => (
              <div key={key} className="bg-gray-50 rounded-lg p-3 text-center">
                <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                  {key.replace(/_/g, " ")}
                </div>
                <div className="text-xl font-bold text-gray-900">
                  {(value as number).toFixed(4)}
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* Model Info */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">Model Details</h4>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">Task Type:</span>
            <span className="font-semibold">Classification</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Target Column:</span>
            <span className="font-semibold">{result.target_column || "N/A"}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Classes:</span>
            <span className="font-semibold">{result.n_classes}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Iterations:</span>
            <span className="font-semibold">{result.n_iterations || "N/A"}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Parameters:</span>
            <span className="font-semibold">{result.architecture?.total_params?.toLocaleString() || "N/A"}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Model ID:</span>
            <code className="text-xs font-mono bg-gray-100 px-1.5 py-0.5 rounded">
              {result.model_id?.slice(0, 20)}...
            </code>
          </div>
        </div>
      </div>
    </div>
  );
}

// ======================== Network Architecture Tab ========================

function ArchitectureTab({ result }: { result: any }) {
  const arch = result.architecture || {};
  const hiddenLayers = arch.hidden_layers || [];
  const inputSize = arch.input_size || result.n_features || 0;
  const outputSize = arch.output_size || result.n_classes || 0;
  const activation = result.metadata?.activation || arch.activation || "relu";

  // Build layer list for visualization
  const allLayers = [
    { label: "Input", size: inputSize, type: "input" },
    ...hiddenLayers.map((size: number, i: number) => ({
      label: `Hidden ${i + 1}`,
      size,
      type: "hidden",
    })),
    { label: "Output", size: outputSize, type: "output" },
  ];

  const maxNeurons = Math.max(...allLayers.map((l: any) => l.size), 1);

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-violet-200 bg-violet-50 px-4 py-3">
        <Network className="mt-0.5 h-5 w-5 shrink-0 text-violet-600" />
        <p className="text-sm text-violet-800">
          <span className="font-semibold">Network Architecture</span> — Your neural network
          has {allLayers.length} layers with {arch.total_params?.toLocaleString() || "?"} trainable
          parameters. Each connection is a weight the model learned during training.
        </p>
      </div>

      {/* Visual Network Diagram */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h4 className="text-sm font-semibold text-gray-900 mb-4">Layer Structure</h4>
        <div className="flex items-center justify-center gap-2 overflow-x-auto py-4">
          {allLayers.map((layer: any, i: number) => {
            const barHeight = Math.max(24, (layer.size / maxNeurons) * 160);
            const isInput = layer.type === "input";
            const isOutput = layer.type === "output";
            const bgColor = isInput
              ? "bg-blue-500"
              : isOutput
              ? "bg-violet-600"
              : "bg-violet-400";

            return (
              <div key={i} className="flex items-center">
                <div className="flex flex-col items-center gap-1 min-w-[60px]">
                  <span className="text-[10px] font-semibold text-gray-500 uppercase">
                    {layer.label}
                  </span>
                  <div
                    className={`${bgColor} rounded-lg flex items-center justify-center transition-all w-12`}
                    style={{ height: barHeight }}
                  >
                    <span className="text-white text-xs font-bold">{layer.size}</span>
                  </div>
                  {layer.type === "hidden" && (
                    <span className="text-[10px] text-gray-400">{activation}</span>
                  )}
                  {layer.type === "output" && (
                    <span className="text-[10px] text-gray-400">softmax</span>
                  )}
                </div>
                {i < allLayers.length - 1 && (
                  <div className="flex flex-col items-center mx-1">
                    <div className="w-6 h-0.5 bg-gray-300" />
                    <span className="text-[9px] text-gray-400 mt-0.5">
                      {layer.size * allLayers[i + 1].size}
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
        <p className="text-xs text-gray-500 text-center mt-2">
          Numbers inside bars = neurons per layer. Numbers on connections = weights between layers.
        </p>
      </div>

      {/* Parameter Breakdown */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">Parameter Breakdown</h4>
        <div className="space-y-2">
          {allLayers.slice(1).map((layer: any, i: number) => {
            const prevSize = allLayers[i].size;
            const weights = prevSize * layer.size;
            const biases = layer.size;
            const total = weights + biases;
            const pct = arch.total_params > 0 ? (total / arch.total_params) * 100 : 0;

            return (
              <div key={i} className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-700 font-medium">
                    {allLayers[i].label} → {layer.label}
                  </span>
                  <span className="font-semibold text-gray-900">
                    {total.toLocaleString()} params ({pct.toFixed(1)}%)
                  </span>
                </div>
                <div className="h-2 rounded-full bg-gray-100 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-violet-400 transition-all duration-500"
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <p className="text-xs text-gray-500">
                  {weights.toLocaleString()} weights + {biases.toLocaleString()} biases
                </p>
              </div>
            );
          })}
        </div>
        <div className="mt-3 pt-3 border-t border-gray-200 flex justify-between text-sm font-semibold">
          <span className="text-gray-700">Total Parameters</span>
          <span className="text-violet-700">{arch.total_params?.toLocaleString() || "N/A"}</span>
        </div>
      </div>

      {/* Learning insight */}
      <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
        <h4 className="text-sm font-semibold text-amber-900 mb-2 flex items-center gap-2">
          <Brain className="w-4 h-4" /> Learning Insight
        </h4>
        <p className="text-sm text-amber-800">
          More parameters = more capacity to learn complex patterns, but also more risk of
          overfitting. Your network has <strong>{arch.total_params?.toLocaleString()}</strong> parameters
          learning from <strong>{result.training_samples?.toLocaleString()}</strong> samples.
          {arch.total_params > result.training_samples * 10
            ? " That's a lot of parameters per sample — watch for overfitting!"
            : " The parameter-to-sample ratio looks reasonable."}
        </p>
      </div>
    </div>
  );
}

// ======================== Training Progress (Loss Curve) Tab ========================

function LossCurveTab({
  data,
  lossCurve,
  validationScores,
}: {
  data: any;
  lossCurve?: number[];
  validationScores?: number[];
}) {
  const lossValues = lossCurve || data?.loss_values || [];
  const valScores = validationScores || data?.validation_scores || [];

  // SVG chart dimensions
  const W = 600;
  const H = 260;
  const PAD = { top: 20, right: 20, bottom: 35, left: 55 };
  const chartW = W - PAD.left - PAD.right;
  const chartH = H - PAD.top - PAD.bottom;

  const lossPath = useMemo(() => {
    if (!lossValues.length) return "";
    const maxLoss = Math.max(...lossValues);
    const minLoss = Math.min(...lossValues);
    const range = maxLoss - minLoss || 1;

    return lossValues
      .map((v: number, i: number) => {
        const x = PAD.left + (i / (lossValues.length - 1)) * chartW;
        const y = PAD.top + (1 - (v - minLoss) / range) * chartH;
        return `${i === 0 ? "M" : "L"}${x},${y}`;
      })
      .join(" ");
  }, [lossValues, chartW, chartH]);

  const valPath = useMemo(() => {
    if (!valScores.length) return "";
    const maxVal = Math.max(...valScores, 0.01);
    const minVal = Math.min(...valScores, 0);
    const range = maxVal - minVal || 1;

    return valScores
      .map((v: number, i: number) => {
        const x = PAD.left + (i / (valScores.length - 1)) * chartW;
        const y = PAD.top + (1 - (v - minVal) / range) * chartH;
        return `${i === 0 ? "M" : "L"}${x},${y}`;
      })
      .join(" ");
  }, [valScores, chartW, chartH]);

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-violet-200 bg-violet-50 px-4 py-3">
        <TrendingDown className="mt-0.5 h-5 w-5 shrink-0 text-violet-600" />
        <p className="text-sm text-violet-800">
          <span className="font-semibold">Training Progress</span> — Watch how the model learned
          over time. The loss should decrease as the network adjusts its weights to make better predictions.
        </p>
      </div>

      {/* Convergence Status */}
      <div className={`rounded-lg border p-4 ${
        data?.convergence === "converged"
          ? "border-green-200 bg-green-50"
          : "border-amber-200 bg-amber-50"
      }`}>
        <div className="flex items-center gap-2 mb-1">
          {data?.convergence === "converged" ? (
            <CheckCircle className="w-4 h-4 text-green-600" />
          ) : (
            <AlertTriangle className="w-4 h-4 text-amber-600" />
          )}
          <span className={`text-sm font-semibold ${
            data?.convergence === "converged" ? "text-green-800" : "text-amber-800"
          }`}>
            {data?.convergence === "converged" ? "Converged" : "Reached Max Iterations"}
          </span>
        </div>
        <p className={`text-sm ${
          data?.convergence === "converged" ? "text-green-700" : "text-amber-700"
        }`}>
          {data?.convergence_message}
        </p>
      </div>

      {/* Loss stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white border border-gray-200 rounded-lg p-4 text-center">
          <div className="text-xs text-gray-500 mb-1">Initial Loss</div>
          <div className="text-lg font-bold text-red-600">{data?.initial_loss?.toFixed(4)}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4 text-center">
          <div className="text-xs text-gray-500 mb-1">Final Loss</div>
          <div className="text-lg font-bold text-green-600">{data?.final_loss?.toFixed(4)}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4 text-center">
          <div className="text-xs text-gray-500 mb-1">Reduction</div>
          <div className="text-lg font-bold text-violet-600">{data?.loss_reduction_pct?.toFixed(1)}%</div>
        </div>
      </div>

      {/* Loss Curve Chart */}
      {lossValues.length > 1 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">Loss Over Iterations</h4>
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 280 }}>
            {/* Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map((frac) => (
              <line
                key={frac}
                x1={PAD.left}
                y1={PAD.top + frac * chartH}
                x2={PAD.left + chartW}
                y2={PAD.top + frac * chartH}
                stroke="#f1f5f9"
                strokeWidth={1}
              />
            ))}

            {/* Loss curve */}
            <path d={lossPath} fill="none" stroke="#8b5cf6" strokeWidth={2.5} strokeLinejoin="round" />

            {/* Validation scores */}
            {valPath && (
              <path d={valPath} fill="none" stroke="#10b981" strokeWidth={2} strokeDasharray="6 3" strokeLinejoin="round" />
            )}

            {/* Axis labels */}
            <text x={PAD.left + chartW / 2} y={H - 4} textAnchor="middle" className="text-[11px]" fill="#64748b">
              Iteration
            </text>
            <text x={12} y={PAD.top + chartH / 2} textAnchor="middle" className="text-[11px]" fill="#64748b"
              transform={`rotate(-90, 12, ${PAD.top + chartH / 2})`}>
              Loss
            </text>

            {/* Legend */}
            <circle cx={PAD.left + chartW - 120} cy={PAD.top + 10} r={4} fill="#8b5cf6" />
            <text x={PAD.left + chartW - 112} y={PAD.top + 14} className="text-[10px]" fill="#64748b">Training Loss</text>
            {valPath && (
              <>
                <circle cx={PAD.left + chartW - 120} cy={PAD.top + 26} r={4} fill="#10b981" />
                <text x={PAD.left + chartW - 112} y={PAD.top + 30} className="text-[10px]" fill="#64748b">Validation Score</text>
              </>
            )}
          </svg>
        </div>
      )}

      {/* Learning insight */}
      <div className="rounded-lg border border-blue-200 bg-blue-50 p-4">
        <h4 className="text-sm font-semibold text-blue-900 mb-2 flex items-center gap-2">
          <Brain className="w-4 h-4" /> What This Tells You
        </h4>
        <ul className="text-sm text-blue-800 space-y-1 list-disc list-inside">
          <li>A <strong>smooth, decreasing</strong> loss curve means the network is learning steadily</li>
          <li>A <strong>jagged</strong> curve may mean the learning rate is too high</li>
          <li>If loss <strong>plateaus early</strong>, the network may need more capacity (layers/neurons)</li>
          <li>If training loss drops but validation doesn't, that's <strong>overfitting</strong></li>
        </ul>
      </div>
    </div>
  );
}

// ======================== Class Distribution Tab ========================

function ClassDistributionTab({ data }: { data: any }) {
  const classes = data?.classes || [];

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-violet-200 bg-violet-50 px-4 py-3">
        <BarChart3 className="mt-0.5 h-5 w-5 shrink-0 text-violet-600" />
        <p className="text-sm text-violet-800">
          <span className="font-semibold">Class Distribution</span> — How many samples does
          each class have? Imbalanced classes can trick the model into favoring the majority class.
        </p>
      </div>

      {/* Balance indicator */}
      <div className={`rounded-lg border p-4 ${
        data?.is_balanced
          ? "border-green-200 bg-green-50"
          : "border-amber-200 bg-amber-50"
      }`}>
        <div className="flex items-center gap-2 mb-1">
          {data?.is_balanced ? (
            <CheckCircle className="w-4 h-4 text-green-600" />
          ) : (
            <AlertTriangle className="w-4 h-4 text-amber-600" />
          )}
          <span className={`text-sm font-semibold ${
            data?.is_balanced ? "text-green-800" : "text-amber-800"
          }`}>
            {data?.is_balanced ? "Balanced Dataset" : "Imbalanced Dataset"}
          </span>
        </div>
        <p className={`text-sm ${data?.is_balanced ? "text-green-700" : "text-amber-700"}`}>
          {data?.balance_message}
        </p>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white border border-gray-200 rounded-lg p-4 text-center">
          <div className="text-xs text-gray-500 mb-1">Total Samples</div>
          <div className="text-lg font-bold text-gray-900">{data?.total_samples?.toLocaleString()}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4 text-center">
          <div className="text-xs text-gray-500 mb-1">Classes</div>
          <div className="text-lg font-bold text-gray-900">{data?.n_classes}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4 text-center">
          <div className="text-xs text-gray-500 mb-1">Imbalance Ratio</div>
          <div className="text-lg font-bold text-gray-900">{data?.imbalance_ratio}:1</div>
        </div>
      </div>

      {/* Bar chart */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">
          Class Breakdown ({data?.target_column})
        </h4>
        <div className="space-y-3">
          {classes.map((cls: any, i: number) => (
            <div key={cls.name} className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="text-gray-700 font-medium">{cls.name}</span>
                <span className="font-semibold text-gray-900">
                  {cls.count.toLocaleString()} ({cls.percentage}%)
                </span>
              </div>
              <div className="h-4 rounded-full bg-gray-100 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${cls.bar_width_pct}%`,
                    backgroundColor: `hsl(${260 + i * 30}, 60%, 55%)`,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Tip */}
      <div className="rounded-lg border border-blue-200 bg-blue-50 p-4">
        <h4 className="text-sm font-semibold text-blue-900 mb-2 flex items-center gap-2">
          <Brain className="w-4 h-4" /> Why This Matters
        </h4>
        <p className="text-sm text-blue-800">
          If one class has 95% of samples, the model can get 95% accuracy by always predicting
          that class — without learning anything useful! Look at <strong>precision</strong> and{" "}
          <strong>recall per class</strong> to catch this. For imbalanced data, consider
          oversampling the minority class or using class weights.
        </p>
      </div>
    </div>
  );
}

// ======================== Metric Explainer Tab ========================

function MetricExplainerTab({ data }: { data: any }) {
  const metrics = data?.metrics || [];
  const colorMap: Record<string, string> = {
    green: "bg-green-50 border-green-200",
    blue: "bg-blue-50 border-blue-200",
    orange: "bg-orange-50 border-orange-200",
    purple: "bg-purple-50 border-purple-200",
    yellow: "bg-yellow-50 border-yellow-200",
    red: "bg-red-50 border-red-200",
  };
  const barColorMap: Record<string, string> = {
    green: "bg-green-500",
    blue: "bg-blue-500",
    orange: "bg-orange-500",
    purple: "bg-purple-500",
    yellow: "bg-yellow-500",
    red: "bg-red-500",
  };

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-purple-200 bg-purple-50 px-4 py-3">
        <Gauge className="mt-0.5 h-5 w-5 shrink-0 text-purple-600" />
        <p className="text-sm text-purple-800">
          <span className="font-semibold">Understanding your metrics</span> — Each metric
          tells a different part of the story about your neural network's performance.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {metrics.map((m: any) => (
          <div
            key={m.metric}
            className={`rounded-lg border p-4 space-y-2 ${colorMap[m.color] || "bg-gray-50 border-gray-200"}`}
          >
            <div className="flex justify-between items-center">
              <h4 className="font-semibold text-gray-900">{m.metric}</h4>
              {m.value_pct != null ? (
                <span className="text-lg font-bold text-gray-900">{m.value_pct}%</span>
              ) : (
                <span className="text-lg font-bold text-gray-900">{m.value}</span>
              )}
            </div>
            {m.value_pct != null && (
              <div className="h-2 rounded-full bg-gray-200 overflow-hidden">
                <div
                  className={`h-full rounded-full ${barColorMap[m.color] || "bg-gray-500"}`}
                  style={{ width: `${Math.min(m.value_pct, 100)}%` }}
                />
              </div>
            )}
            <p className="text-sm text-gray-700">{m.analogy}</p>
            <p className="text-xs text-gray-500 italic">{m.when_useful}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ======================== Quiz Tab ========================

function QuizTab({ questions }: { questions: any[] }) {
  const [currentQ, setCurrentQ] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [answered, setAnswered] = useState(false);
  const [score, setScore] = useState(0);
  const [answers, setAnswers] = useState<(number | null)[]>([]);
  const [showResults, setShowResults] = useState(false);

  if (!questions.length) return null;

  const question = questions[currentQ];
  const isCorrect = selectedAnswer === question.correct_answer;

  const handleSelect = (idx: number) => {
    if (answered) return;
    setSelectedAnswer(idx);
    setAnswered(true);
    const correct = idx === question.correct_answer;
    if (correct) setScore((s) => s + 1);
    setAnswers((a) => [...a, idx]);
  };

  const handleNext = () => {
    if (currentQ + 1 >= questions.length) {
      setShowResults(true);
    } else {
      setCurrentQ((q) => q + 1);
      setSelectedAnswer(null);
      setAnswered(false);
    }
  };

  const handleRetry = () => {
    setCurrentQ(0);
    setSelectedAnswer(null);
    setAnswered(false);
    setScore(0);
    setAnswers([]);
    setShowResults(false);
  };

  if (showResults) {
    const pct = Math.round((score / questions.length) * 100);
    return (
      <div className="space-y-6">
        <div className="text-center py-8">
          <Trophy
            className={`w-16 h-16 mx-auto mb-4 ${
              pct >= 80 ? "text-violet-500" : pct >= 50 ? "text-yellow-500" : "text-red-500"
            }`}
          />
          <h3 className="text-2xl font-bold text-gray-900">
            {score} / {questions.length}
          </h3>
          <p className="text-gray-600 mt-1">
            {pct >= 80
              ? "Excellent! You understand neural networks well!"
              : pct >= 50
              ? "Good job! Review the topics you missed."
              : "Keep learning! Neural networks have many concepts to master."}
          </p>
          <button
            onClick={handleRetry}
            className="mt-4 px-6 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 transition-colors text-sm font-medium"
          >
            Try Again
          </button>
        </div>

        {/* Review */}
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-gray-700">Review</h4>
          {questions.map((q: any, i: number) => {
            const userAns = answers[i];
            const correct = userAns === q.correct_answer;
            return (
              <div key={i} className={`rounded-lg border p-3 ${correct ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"}`}>
                <div className="flex items-start gap-2">
                  {correct ? (
                    <CheckCircle className="w-4 h-4 text-green-600 mt-0.5 shrink-0" />
                  ) : (
                    <XCircle className="w-4 h-4 text-red-600 mt-0.5 shrink-0" />
                  )}
                  <div>
                    <p className="text-sm font-medium text-gray-900">{q.question}</p>
                    <p className="text-xs text-gray-600 mt-1">
                      Your answer: {q.options[userAns ?? 0]} {!correct && `| Correct: ${q.options[q.correct_answer]}`}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">{q.explanation}</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-violet-200 bg-violet-50 px-4 py-3">
        <HelpCircle className="mt-0.5 h-5 w-5 shrink-0 text-violet-600" />
        <p className="text-sm text-violet-800">
          <span className="font-semibold">Test your knowledge</span> about Neural Networks!
          Answer {questions.length} questions to check your understanding.
        </p>
      </div>

      {/* Progress dots */}
      <div className="flex items-center justify-center gap-2">
        {questions.map((_: any, i: number) => (
          <div
            key={i}
            className={`w-2.5 h-2.5 rounded-full transition-all ${
              i === currentQ
                ? "bg-violet-500 scale-125"
                : i < answers.length
                ? answers[i] === questions[i].correct_answer
                  ? "bg-green-400"
                  : "bg-red-400"
                : "bg-gray-300"
            }`}
          />
        ))}
      </div>

      {/* Question */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <p className="text-xs text-gray-400 mb-2">
          Question {currentQ + 1} of {questions.length}
        </p>
        <p className="text-base font-medium text-gray-900 mb-4">{question.question}</p>

        <div className="space-y-2">
          {question.options.map((opt: string, idx: number) => {
            let style = "border-gray-200 bg-white hover:bg-gray-50 text-gray-700";
            if (answered) {
              if (idx === question.correct_answer) {
                style = "border-green-300 bg-green-50 text-green-800";
              } else if (idx === selectedAnswer && !isCorrect) {
                style = "border-red-300 bg-red-50 text-red-800";
              } else {
                style = "border-gray-200 bg-gray-50 text-gray-400";
              }
            } else if (idx === selectedAnswer) {
              style = "border-violet-400 bg-violet-50 text-violet-700";
            }

            return (
              <button
                key={idx}
                onClick={() => handleSelect(idx)}
                disabled={answered}
                className={`w-full text-left px-4 py-3 rounded-lg border text-sm transition-colors ${style} disabled:cursor-default`}
              >
                <span className="font-medium mr-2">
                  {String.fromCharCode(65 + idx)}.
                </span>
                {opt}
              </button>
            );
          })}
        </div>

        {answered && (
          <div className={`mt-4 p-3 rounded-lg text-sm ${isCorrect ? "bg-green-50 text-green-800" : "bg-red-50 text-red-800"}`}>
            <p className="font-semibold mb-1">{isCorrect ? "Correct!" : "Not quite."}</p>
            <p>{question.explanation}</p>
          </div>
        )}

        {answered && (
          <button
            onClick={handleNext}
            className="mt-4 px-4 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 transition-colors text-sm font-medium"
          >
            {currentQ + 1 >= questions.length ? "See Results" : "Next Question"}
          </button>
        )}
      </div>
    </div>
  );
}

// ======================== How It Works Tab ========================

function HowItWorksTab({ architecture, activation }: { architecture?: any; activation?: string }) {
  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-violet-200 bg-violet-50 px-4 py-3">
        <Cog className="mt-0.5 h-5 w-5 shrink-0 text-violet-600" />
        <div className="text-sm text-violet-800 space-y-2">
          <p className="font-semibold">How MLP Classifier Works</p>
          <ol className="list-decimal list-inside space-y-1">
            <li><strong>Forward pass:</strong> Input features pass through layers of neurons, each applying weights + bias + activation function</li>
            <li><strong>Activation:</strong> Non-linear function ({activation || "relu"}) allows learning complex, curved decision boundaries</li>
            <li><strong>Output layer:</strong> Softmax converts final values into class probabilities (they sum to 1)</li>
            <li><strong>Loss calculation:</strong> Cross-entropy measures how far predictions are from true labels</li>
            <li><strong>Backpropagation:</strong> Calculates how much each weight contributed to the error</li>
            <li><strong>Weight update:</strong> Optimizer (like Adam) adjusts weights to reduce the loss</li>
            <li><strong>Repeat:</strong> Steps 1-6 repeat for many iterations until loss converges</li>
          </ol>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Brain className="w-4 h-4 text-violet-600" />
            Universal Approximator
          </h4>
          <p className="text-sm text-gray-600">
            With enough neurons, an MLP can learn any function. This is called the
            Universal Approximation Theorem. But more neurons need more data to
            avoid memorizing instead of learning.
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Zap className="w-4 h-4 text-amber-600" />
            Activation Functions
          </h4>
          <p className="text-sm text-gray-600">
            Without activation functions, stacking layers is pointless — the result
            is still just a linear function. ReLU (max(0, x)) is the most popular
            because it's fast and avoids the vanishing gradient problem.
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Target className="w-4 h-4 text-red-600" />
            Overfitting Risk
          </h4>
          <p className="text-sm text-gray-600">
            Neural networks can memorize training data perfectly but fail on new
            data. Use early stopping, L2 regularization (alpha), or reduce network
            size to combat this.
          </p>
        </div>
      </div>

      <div className="rounded-lg border border-gray-200 bg-white p-4">
        <h4 className="font-semibold text-gray-900 mb-2">Key Hyperparameters</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
          <div className="flex gap-2">
            <span className="font-semibold text-violet-700 shrink-0">hidden_layer_sizes:</span>
            <span className="text-gray-600">Number and size of hidden layers (e.g., "100, 50" = two layers).</span>
          </div>
          <div className="flex gap-2">
            <span className="font-semibold text-violet-700 shrink-0">activation:</span>
            <span className="text-gray-600">Non-linear function applied to each neuron (relu, tanh, logistic).</span>
          </div>
          <div className="flex gap-2">
            <span className="font-semibold text-violet-700 shrink-0">learning_rate:</span>
            <span className="text-gray-600">Step size for weight updates. Too high = unstable, too low = slow.</span>
          </div>
          <div className="flex gap-2">
            <span className="font-semibold text-violet-700 shrink-0">alpha:</span>
            <span className="text-gray-600">L2 regularization strength. Higher values penalize large weights.</span>
          </div>
          <div className="flex gap-2">
            <span className="font-semibold text-violet-700 shrink-0">early_stopping:</span>
            <span className="text-gray-600">Stop training when validation score stops improving.</span>
          </div>
          <div className="flex gap-2">
            <span className="font-semibold text-violet-700 shrink-0">max_iter:</span>
            <span className="text-gray-600">Maximum number of training iterations (epochs).</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MLPClassifierExplorer;
