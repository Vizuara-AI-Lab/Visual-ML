/**
 * CNNClassifierExplorer - Interactive learning for CNN/MLP Classifier node.
 * Tabs: Results | Architecture | Training Curves | Feature Importance | Quiz
 */

import { useState, type ReactNode } from "react";
import {
  ClipboardList,
  Layers,
  TrendingUp,
  Grid3X3,
  HelpCircle,
  Info,
  AlertTriangle,
  CheckCircle,
} from "lucide-react";
import { QuizTab } from "./ImageDatasetExplorer";

// --- Main Explorer ---

interface CNNClassifierExplorerProps {
  result: any;
  renderResults: () => ReactNode;
}

type Tab = "results" | "architecture" | "training" | "importance" | "quiz";

export const CNNClassifierExplorer = ({
  result,
  renderResults,
}: CNNClassifierExplorerProps) => {
  const [activeTab, setActiveTab] = useState<Tab>("results");

  const tabs: { id: Tab; label: string; icon: any; available: boolean }[] = [
    { id: "results", label: "Results", icon: ClipboardList, available: true },
    {
      id: "architecture",
      label: "Architecture",
      icon: Layers,
      available: !!result.architecture_diagram,
    },
    {
      id: "training",
      label: "Training Curves",
      icon: TrendingUp,
      available: !!result.training_analysis?.has_data,
    },
    {
      id: "importance",
      label: "Feature Importance",
      icon: Grid3X3,
      available: !!result.feature_importance?.heatmap?.length,
    },
    {
      id: "quiz",
      label: "Quiz",
      icon: HelpCircle,
      available: !!result.quiz_questions && result.quiz_questions.length > 0,
    },
  ];

  return (
    <div className="space-y-4">
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
                    ? "border-blue-600 text-blue-600"
                    : "border-transparent text-gray-500 hover:text-gray-700"
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
      </div>

      {activeTab === "results" && renderResults()}
      {activeTab === "architecture" && result.architecture_diagram && (
        <ArchitectureTab data={result.architecture_diagram} />
      )}
      {activeTab === "training" && result.training_analysis && (
        <TrainingTab data={result.training_analysis} />
      )}
      {activeTab === "importance" && result.feature_importance && (
        <FeatureImportanceTab data={result.feature_importance} />
      )}
      {activeTab === "quiz" && result.quiz_questions && (
        <QuizTab questions={result.quiz_questions} />
      )}
    </div>
  );
};

// --- Architecture Tab ---

function ArchitectureTab({ data }: { data: any }) {
  return (
    <div className="space-y-4">
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-purple-900">
          🏗️ Network Architecture
        </h3>
        <p className="text-xs text-purple-700 mt-1">
          Your neural network has {data.layers.length} layers with{" "}
          {data.total_params.toLocaleString()} learnable parameters.
        </p>
      </div>

      {/* Visual architecture diagram */}
      <div className="bg-white border border-gray-200 rounded-lg p-4 overflow-x-auto">
        <div className="flex items-center gap-3 min-w-max justify-center">
          {data.layers.map((layer: any, i: number) => {
            const maxNeurons = Math.max(
              ...data.layers.map((l: any) => l.size)
            );
            const barHeight = Math.max(
              20,
              Math.min(120, (layer.size / maxNeurons) * 120)
            );

            const bgColor =
              layer.type === "input"
                ? "bg-gradient-to-b from-blue-400 to-blue-600"
                : layer.type === "output"
                  ? "bg-gradient-to-b from-green-400 to-green-600"
                  : "bg-gradient-to-b from-purple-400 to-purple-600";

            return (
              <div key={i} className="flex items-center gap-3">
                <div className="flex flex-col items-center">
                  {/* Layer bar */}
                  <div
                    className={`${bgColor} rounded-lg flex items-center justify-center text-white text-xs font-bold shadow-md transition-all hover:scale-105`}
                    style={{
                      width: layer.type === "input" || layer.type === "output" ? 60 : 50,
                      height: barHeight,
                      minHeight: 30,
                    }}
                    title={layer.description}
                  >
                    {layer.size}
                  </div>

                  {/* Layer label */}
                  <div className="mt-2 text-center">
                    <div className="text-[10px] font-semibold text-gray-700">
                      {layer.type === "input"
                        ? "Input"
                        : layer.type === "output"
                          ? "Output"
                          : `Hidden ${i}`}
                    </div>
                    {layer.activation && (
                      <div className="text-[9px] text-gray-400">
                        {layer.activation}
                      </div>
                    )}
                    {layer.params && (
                      <div className="text-[9px] text-gray-400">
                        {layer.params.toLocaleString()} params
                      </div>
                    )}
                  </div>
                </div>

                {/* Arrow */}
                {i < data.layers.length - 1 && (
                  <div className="text-gray-300 text-lg">→</div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Layer details table */}
      <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="text-left px-3 py-2 text-xs text-gray-600">
                Layer
              </th>
              <th className="text-center px-3 py-2 text-xs text-gray-600">
                Neurons
              </th>
              <th className="text-center px-3 py-2 text-xs text-gray-600">
                Activation
              </th>
              <th className="text-center px-3 py-2 text-xs text-gray-600">
                Parameters
              </th>
            </tr>
          </thead>
          <tbody>
            {data.layers.map((layer: any, i: number) => (
              <tr key={i} className="border-t hover:bg-gray-50">
                <td className="px-3 py-2">
                  <span className="font-medium">{layer.name}</span>
                </td>
                <td className="px-3 py-2 text-center font-mono">
                  {layer.size}
                </td>
                <td className="px-3 py-2 text-center text-xs">
                  {layer.activation || "—"}
                </td>
                <td className="px-3 py-2 text-center font-mono text-xs">
                  {layer.params ? layer.params.toLocaleString() : "—"}
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot className="bg-gray-50">
            <tr className="border-t font-semibold">
              <td className="px-3 py-2" colSpan={3}>
                Total Parameters
              </td>
              <td className="px-3 py-2 text-center font-mono">
                {data.total_params.toLocaleString()}
              </td>
            </tr>
          </tfoot>
        </table>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <p className="text-xs text-blue-700">
            <strong>Parameters</strong> are the learnable weights — each
            connection between neurons has a weight, plus each neuron has a
            bias. More parameters = more capacity to learn complex patterns,
            but also more risk of overfitting.
          </p>
        </div>
      </div>
    </div>
  );
}

// --- Training Curves Tab ---

function TrainingTab({ data }: { data: any }) {
  const lossCurve = data.loss_values || [];
  const valScores = data.validation_scores || [];
  const maxLoss = Math.max(...lossCurve, 0.01);
  const maxVal = valScores.length > 0 ? Math.max(...valScores, 0.01) : null;

  return (
    <div className="space-y-4">
      <div className="bg-orange-50 border border-orange-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-orange-900">
          📉 Training Curves
        </h3>
        <p className="text-xs text-orange-700 mt-1">
          {data.convergence_message}
        </p>
      </div>

      {/* Loss curve */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-800 mb-3">
          Training Loss
        </h4>
        <div className="flex items-end gap-[1px] h-40">
          {lossCurve.map((loss: number, i: number) => {
            const height = (loss / maxLoss) * 100;
            return (
              <div
                key={i}
                className="flex-1 bg-red-400 hover:bg-red-600 transition-colors rounded-t min-w-[2px]"
                style={{ height: `${height}%` }}
                title={`Iteration ${i + 1}: loss = ${loss.toFixed(6)}`}
              />
            );
          })}
        </div>
        <div className="flex justify-between text-[10px] text-gray-500 mt-1">
          <span>Iteration 1</span>
          <span>Loss →</span>
          <span>{lossCurve.length}</span>
        </div>
      </div>

      {/* Validation scores */}
      {valScores.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-800 mb-3">
            Validation Accuracy
          </h4>
          <div className="flex items-end gap-[1px] h-32">
            {valScores.map((score: number, i: number) => {
              const height = (score / (maxVal || 1)) * 100;
              return (
                <div
                  key={i}
                  className="flex-1 bg-green-400 hover:bg-green-600 transition-colors rounded-t min-w-[2px]"
                  style={{ height: `${height}%` }}
                  title={`Iteration ${i + 1}: accuracy = ${(score * 100).toFixed(1)}%`}
                />
              );
            })}
          </div>
          <div className="flex justify-between text-[10px] text-gray-500 mt-1">
            <span>Iteration 1</span>
            <span>Accuracy →</span>
            <span>{valScores.length}</span>
          </div>
        </div>
      )}

      {/* Summary stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-xs text-gray-500">Initial Loss</div>
          <div className="text-lg font-bold text-red-600">
            {data.initial_loss}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-xs text-gray-500">Final Loss</div>
          <div className="text-lg font-bold text-green-600">
            {data.final_loss}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-xs text-gray-500">Reduction</div>
          <div className="text-lg font-bold text-blue-600">
            {data.loss_reduction_pct}%
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-xs text-gray-500">Convergence</div>
          <div className="text-lg font-bold">
            {data.convergence === "converged" ? (
              <span className="text-green-600">✓</span>
            ) : (
              <span className="text-orange-600">⚠️</span>
            )}
          </div>
        </div>
      </div>

      {data.overfitting_warning && (
        <div className="bg-orange-50 border border-orange-200 rounded-lg p-3 flex items-start gap-2">
          <AlertTriangle className="w-4 h-4 text-orange-600 mt-0.5 shrink-0" />
          <p className="text-xs text-orange-700">{data.overfitting_warning}</p>
        </div>
      )}
    </div>
  );
}

// --- Feature Importance Tab ---

function FeatureImportanceTab({ data }: { data: any }) {
  const heatmap = data.heatmap;
  const width = data.width;
  const height = data.height;

  const canvasRef = (canvas: HTMLCanvasElement | null) => {
    if (!canvas || !heatmap || heatmap.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const imageData = ctx.createImageData(width, height);

    for (let i = 0; i < width * height && i < heatmap.length; i++) {
      const val = heatmap[i]; // Already 0-1
      // Use a blue-to-red heatmap
      const r = Math.round(val * 255);
      const g = Math.round((1 - Math.abs(val - 0.5) * 2) * 255 * 0.5);
      const b = Math.round((1 - val) * 100);
      imageData.data[i * 4] = r;
      imageData.data[i * 4 + 1] = g;
      imageData.data[i * 4 + 2] = b;
      imageData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  };

  return (
    <div className="space-y-4">
      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-indigo-900">
          🔥 Feature Importance Heatmap
        </h3>
        <p className="text-xs text-indigo-700 mt-1">{data.description}</p>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-6 flex flex-col items-center">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          style={{
            width: Math.max(width * 8, 200),
            height: Math.max(height * 8, 200),
            imageRendering: "pixelated",
            borderRadius: 8,
            border: "2px solid #e5e7eb",
          }}
        />

        {/* Legend */}
        <div className="flex items-center gap-2 mt-4">
          <span className="text-xs text-gray-500">Low Importance</span>
          <div
            className="h-3 w-32 rounded-full"
            style={{
              background:
                "linear-gradient(to right, rgb(0,0,100), rgb(128,128,0), rgb(255,0,0))",
            }}
          />
          <span className="text-xs text-gray-500">High Importance</span>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <p className="text-xs text-blue-700">
            This heatmap shows the sum of absolute weights from the first layer
            of the neural network. Brighter (redder) pixels had more influence
            on the model's predictions. For digit datasets, you'll typically see
            the center lit up — that's where the strokes are, and where the
            model pays attention.
          </p>
        </div>
      </div>
    </div>
  );
}
