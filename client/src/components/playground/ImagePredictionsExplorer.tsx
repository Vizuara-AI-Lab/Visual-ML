/**
 * ImagePredictionsExplorer - Interactive learning for Image Predictions node.
 * Tabs: Results | Prediction Gallery | Confusion Analysis | Confidence | Quiz
 */

import { useState, type ReactNode } from "react";
import {
  ClipboardList,
  Image,
  Grid3X3,
  BarChart3,
  HelpCircle,
  Info,
  CheckCircle,
  XCircle,
} from "lucide-react";
import { QuizTab } from "./ImageDatasetExplorer";

// --- Shared image renderer ---
function PixelImage({
  pixels,
  width,
  height,
  size = 64,
  label,
  borderColor,
}: {
  pixels: number[];
  width: number;
  height: number;
  size?: number;
  label?: string;
  borderColor?: string;
}) {
  const canvasRef = (canvas: HTMLCanvasElement | null) => {
    if (!canvas || !pixels || pixels.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const imageData = ctx.createImageData(width, height);
    const maxVal = Math.max(...pixels, 1);
    const minVal = Math.min(...pixels, 0);
    const range = maxVal - minVal || 1;
    for (let i = 0; i < width * height; i++) {
      const val = Math.round(((pixels[i] - minVal) / range) * 255);
      imageData.data[i * 4] = val;
      imageData.data[i * 4 + 1] = val;
      imageData.data[i * 4 + 2] = val;
      imageData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  };

  return (
    <div className="flex flex-col items-center gap-1">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          width: size,
          height: size,
          imageRendering: "pixelated",
          border: borderColor ? `2px solid ${borderColor}` : "1px solid #e5e7eb",
          borderRadius: 4,
        }}
      />
      {label && (
        <span className="text-[10px] text-gray-600 truncate max-w-[100px]">
          {label}
        </span>
      )}
    </div>
  );
}

// --- Main Explorer ---

interface ImagePredictionsExplorerProps {
  result: any;
  renderResults: () => ReactNode;
}

type Tab = "results" | "gallery" | "confusion" | "confidence" | "quiz";

export const ImagePredictionsExplorer = ({
  result,
  renderResults,
}: ImagePredictionsExplorerProps) => {
  const [activeTab, setActiveTab] = useState<Tab>("results");

  const tabs: { id: Tab; label: string; icon: any; available: boolean }[] = [
    { id: "results", label: "Results", icon: ClipboardList, available: true },
    {
      id: "gallery",
      label: "Prediction Gallery",
      icon: Image,
      available: !!result.prediction_gallery && result.prediction_gallery.length > 0,
    },
    {
      id: "confusion",
      label: "Confusion Matrix",
      icon: Grid3X3,
      available: !!result.confusion_analysis,
    },
    {
      id: "confidence",
      label: "Confidence",
      icon: BarChart3,
      available: !!result.confidence_distribution?.has_data,
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
      {activeTab === "gallery" && result.prediction_gallery && (
        <PredictionGalleryTab
          data={result.prediction_gallery}
          metrics={result.metrics_summary}
        />
      )}
      {activeTab === "confusion" && result.confusion_analysis && (
        <ConfusionTab data={result.confusion_analysis} />
      )}
      {activeTab === "confidence" && result.confidence_distribution && (
        <ConfidenceTab data={result.confidence_distribution} />
      )}
      {activeTab === "quiz" && result.quiz_questions && (
        <QuizTab questions={result.quiz_questions} />
      )}
    </div>
  );
};

// --- Prediction Gallery Tab ---

function PredictionGalleryTab({
  data,
  metrics,
}: {
  data: any[];
  metrics?: any;
}) {
  const [filter, setFilter] = useState<"all" | "correct" | "wrong">("all");

  const correct = data.filter((d: any) => d.correct);
  const wrong = data.filter((d: any) => !d.correct);
  const filtered =
    filter === "correct" ? correct : filter === "wrong" ? wrong : data;

  return (
    <div className="space-y-4">
      {/* Metrics summary */}
      {metrics && (
        <div className="grid grid-cols-4 gap-3">
          <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
            <div className="text-xs text-gray-500">Accuracy</div>
            <div className="text-2xl font-bold text-blue-600">
              {(metrics.overall_accuracy * 100).toFixed(1)}%
            </div>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
            <div className="text-xs text-gray-500">Grade</div>
            <div
              className={`text-2xl font-bold ${
                metrics.grade === "A+" || metrics.grade === "A"
                  ? "text-green-600"
                  : metrics.grade === "B"
                    ? "text-blue-600"
                    : metrics.grade === "C"
                      ? "text-yellow-600"
                      : "text-red-600"
              }`}
            >
              {metrics.grade}
            </div>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
            <div className="text-xs text-gray-500">Best Class</div>
            <div className="text-sm font-bold text-green-600 truncate">
              {metrics.best_class?.name}
            </div>
            <div className="text-[10px] text-gray-400">
              F1: {metrics.best_class?.f1}
            </div>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
            <div className="text-xs text-gray-500">Worst Class</div>
            <div className="text-sm font-bold text-red-600 truncate">
              {metrics.worst_class?.name}
            </div>
            <div className="text-[10px] text-gray-400">
              F1: {metrics.worst_class?.f1}
            </div>
          </div>
        </div>
      )}

      {/* Filter buttons */}
      <div className="flex gap-2">
        {[
          { id: "all", label: `All (${data.length})` },
          { id: "correct", label: `✓ Correct (${correct.length})` },
          { id: "wrong", label: `✗ Wrong (${wrong.length})` },
        ].map((f) => (
          <button
            key={f.id}
            onClick={() => setFilter(f.id as any)}
            className={`px-3 py-1.5 text-sm rounded-lg border transition-all ${
              filter === f.id
                ? "bg-blue-600 text-white border-blue-600"
                : "bg-white text-gray-700 border-gray-300 hover:border-blue-400"
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* Image grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
        {filtered.map((pred: any, i: number) => (
          <div
            key={i}
            className={`bg-white border-2 rounded-lg p-3 ${
              pred.correct ? "border-green-200" : "border-red-200"
            }`}
          >
            <div className="flex justify-center mb-2">
              <PixelImage
                pixels={pred.pixels}
                width={pred.width}
                height={pred.height}
                size={72}
                borderColor={pred.correct ? "#22c55e" : "#ef4444"}
              />
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center gap-1 mb-1">
                {pred.correct ? (
                  <CheckCircle className="w-3 h-3 text-green-600" />
                ) : (
                  <XCircle className="w-3 h-3 text-red-600" />
                )}
                <span
                  className={`text-xs font-semibold ${
                    pred.correct ? "text-green-700" : "text-red-700"
                  }`}
                >
                  {pred.correct ? "Correct" : "Wrong"}
                </span>
              </div>
              <div className="text-[10px] text-gray-600">
                True: <strong>{pred.true_label}</strong>
              </div>
              <div className="text-[10px] text-gray-600">
                Pred: <strong>{pred.pred_label}</strong>
              </div>
              {pred.confidence !== undefined && (
                <div className="mt-1">
                  <div className="w-full bg-gray-100 rounded-full h-1.5">
                    <div
                      className={`h-full rounded-full ${
                        pred.correct ? "bg-green-500" : "bg-red-500"
                      }`}
                      style={{ width: `${pred.confidence * 100}%` }}
                    />
                  </div>
                  <div className="text-[9px] text-gray-400">
                    {(pred.confidence * 100).toFixed(0)}% confident
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// --- Confusion Matrix Tab ---

function ConfusionTab({ data }: { data: any }) {
  const [hoveredCell, setHoveredCell] = useState<{
    r: number;
    c: number;
  } | null>(null);

  const cm = data.confusion_matrix;
  const classNames = data.class_names;
  const maxVal = Math.max(
    ...cm.flatMap((row: number[]) => row),
    1
  );

  return (
    <div className="space-y-4">
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-purple-900">
          📊 Confusion Matrix
        </h3>
        <p className="text-xs text-purple-700 mt-1">
          Rows = actual class, Columns = predicted class. Diagonal = correct
          predictions.
        </p>
      </div>

      {/* Matrix grid */}
      <div className="bg-white border border-gray-200 rounded-lg p-4 overflow-x-auto">
        <table className="mx-auto">
          <thead>
            <tr>
              <th className="text-[9px] text-gray-400 text-right pr-2 pb-1">
                Actual ↓ / Pred →
              </th>
              {classNames.map((name: string, j: number) => (
                <th
                  key={j}
                  className="text-[9px] text-gray-600 font-medium px-1 pb-1 text-center"
                >
                  {name.length > 6 ? name.slice(0, 5) + ".." : name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cm.map((row: number[], i: number) => (
              <tr key={i}>
                <td className="text-[9px] text-gray-600 font-medium pr-2 text-right">
                  {classNames[i]?.length > 6
                    ? classNames[i].slice(0, 5) + ".."
                    : classNames[i]}
                </td>
                {row.map((count: number, j: number) => {
                  const isDiag = i === j;
                  const intensity = count / maxVal;
                  const isHovered =
                    hoveredCell?.r === i && hoveredCell?.c === j;

                  return (
                    <td
                      key={j}
                      className={`text-center transition-all ${
                        isHovered ? "ring-2 ring-blue-500" : ""
                      }`}
                      style={{
                        width: 36,
                        height: 32,
                        fontSize: 11,
                        fontWeight: isDiag ? 700 : 400,
                        color: intensity > 0.5 ? "#fff" : "#1f2937",
                        backgroundColor: isDiag
                          ? `rgba(34, 197, 94, ${0.1 + intensity * 0.9})`
                          : `rgba(239, 68, 68, ${intensity * 0.8})`,
                      }}
                      onMouseEnter={() => setHoveredCell({ r: i, c: j })}
                      onMouseLeave={() => setHoveredCell(null)}
                      title={`True: ${classNames[i]}, Pred: ${classNames[j]} — ${count} images`}
                    >
                      {count}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Most confused pairs */}
      {data.most_confused_pairs?.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-800 mb-3">
            ⚠️ Most Confused Pairs
          </h4>
          <div className="space-y-2">
            {data.most_confused_pairs.slice(0, 5).map((pair: any, i: number) => (
              <div
                key={i}
                className="flex items-center gap-3 bg-red-50 border border-red-100 rounded-lg px-3 py-2"
              >
                <span className="text-sm font-medium text-red-800">
                  {pair.true_class}
                </span>
                <span className="text-gray-400">→</span>
                <span className="text-sm font-medium text-red-800">
                  {pair.predicted_class}
                </span>
                <span className="ml-auto text-xs text-red-600 font-mono">
                  {pair.count}×
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Per-class accuracy */}
      {data.per_class_accuracy && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-800 mb-3">
            Per-Class Accuracy
          </h4>
          <div className="space-y-2">
            {data.per_class_accuracy.map((cls: any, i: number) => (
              <div key={i} className="flex items-center gap-3">
                <div className="w-20 text-xs font-medium text-gray-700 truncate">
                  {cls.class_name}
                </div>
                <div className="flex-1 bg-gray-100 rounded-full h-4 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${
                      cls.accuracy >= 0.9
                        ? "bg-green-500"
                        : cls.accuracy >= 0.7
                          ? "bg-yellow-500"
                          : "bg-red-500"
                    }`}
                    style={{ width: `${cls.accuracy * 100}%` }}
                  />
                </div>
                <div className="w-16 text-right text-xs font-mono text-gray-600">
                  {(cls.accuracy * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// --- Confidence Distribution Tab ---

function ConfidenceTab({ data }: { data: any }) {
  const maxCount = Math.max(
    ...data.correct_counts,
    ...data.wrong_counts,
    1
  );

  return (
    <div className="space-y-4">
      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-indigo-900">
          🎯 Prediction Confidence
        </h3>
        <p className="text-xs text-indigo-700 mt-1">{data.insight}</p>
      </div>

      {/* Histogram */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-800 mb-3">
          Confidence Distribution
        </h4>
        <div className="flex items-end gap-1 h-32">
          {data.correct_counts.map((count: number, i: number) => {
            const wrongCount = data.wrong_counts[i] || 0;
            const correctH = (count / maxCount) * 100;
            const wrongH = (wrongCount / maxCount) * 100;

            return (
              <div
                key={i}
                className="flex-1 flex flex-col items-stretch justify-end"
                title={`${data.bins[i]}–${data.bins[i + 1]}: ${count} correct, ${wrongCount} wrong`}
              >
                {wrongH > 0 && (
                  <div
                    className="bg-red-400 rounded-t"
                    style={{ height: `${wrongH}%`, minHeight: wrongCount > 0 ? 2 : 0 }}
                  />
                )}
                {correctH > 0 && (
                  <div
                    className="bg-green-500"
                    style={{
                      height: `${correctH}%`,
                      minHeight: count > 0 ? 2 : 0,
                      borderRadius: wrongH > 0 ? 0 : "4px 4px 0 0",
                    }}
                  />
                )}
              </div>
            );
          })}
        </div>
        <div className="flex justify-between text-[10px] text-gray-500 mt-1">
          <span>0% confidence</span>
          <span>100% confidence</span>
        </div>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-green-50 border border-green-200 rounded-lg p-3 text-center">
          <div className="text-xs text-green-600">Correct Predictions</div>
          <div className="text-xl font-bold text-green-700">
            {(data.correct_mean_confidence * 100).toFixed(1)}%
          </div>
          <div className="text-[10px] text-green-500">avg. confidence</div>
        </div>
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-center">
          <div className="text-xs text-red-600">Wrong Predictions</div>
          <div className="text-xl font-bold text-red-700">
            {(data.wrong_mean_confidence * 100).toFixed(1)}%
          </div>
          <div className="text-[10px] text-red-500">avg. confidence</div>
        </div>
      </div>

      <div className="flex items-center gap-4 text-xs text-gray-500">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 bg-green-500 rounded-sm" /> Correct
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 bg-red-400 rounded-sm" /> Wrong
        </span>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <p className="text-xs text-blue-700">
            <strong>Well-calibrated</strong> models should be more confident
            when they're correct and less confident when wrong. If the model is
            highly confident in wrong predictions, it may be overfitting. The
            gap between green and red bars tells you how well-calibrated the
            model is.
          </p>
        </div>
      </div>
    </div>
  );
}
