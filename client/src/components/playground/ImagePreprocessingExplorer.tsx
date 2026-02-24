/**
 * ImagePreprocessingExplorer - Interactive learning for Image Preprocessing node.
 * Tabs: Results | Before & After | Normalization Lab | Pixel Histograms | Quiz
 */

import { useState, useMemo, type ReactNode } from "react";
import {
  ClipboardList,
  ArrowLeftRight,
  FlaskConical,
  BarChart3,
  HelpCircle,
  Info,
} from "lucide-react";
import { QuizTab } from "./ImageDatasetExplorer";

// --- Pixel Image Renderer (shared) ---
function PixelImage({
  pixels,
  width,
  height,
  size = 64,
  label,
}: {
  pixels: number[];
  width: number;
  height: number;
  size?: number;
  label?: string;
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
          border: "1px solid #e5e7eb",
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

interface ImagePreprocessingExplorerProps {
  result: any;
  renderResults: () => ReactNode;
}

type Tab = "results" | "before_after" | "norm_lab" | "histograms" | "quiz";

export const ImagePreprocessingExplorer = ({
  result,
  renderResults,
}: ImagePreprocessingExplorerProps) => {
  const [activeTab, setActiveTab] = useState<Tab>("results");

  const tabs: { id: Tab; label: string; icon: any; available: boolean }[] = [
    { id: "results", label: "Results", icon: ClipboardList, available: true },
    {
      id: "before_after",
      label: "Before & After",
      icon: ArrowLeftRight,
      available: !!result.before_after_samples && result.before_after_samples.length > 0,
    },
    {
      id: "norm_lab",
      label: "Normalization Lab",
      icon: FlaskConical,
      available: !!result.normalization_comparison,
    },
    {
      id: "histograms",
      label: "Pixel Histograms",
      icon: BarChart3,
      available: !!result.pixel_histograms,
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
      {activeTab === "before_after" && result.before_after_samples && (
        <BeforeAfterTab samples={result.before_after_samples} />
      )}
      {activeTab === "norm_lab" && result.normalization_comparison && (
        <NormLabTab data={result.normalization_comparison} />
      )}
      {activeTab === "histograms" && result.pixel_histograms && (
        <HistogramTab data={result.pixel_histograms} />
      )}
      {activeTab === "quiz" && result.quiz_questions && (
        <QuizTab questions={result.quiz_questions} />
      )}
    </div>
  );
};

// --- Before & After Tab ---

function BeforeAfterTab({ samples }: { samples: any[] }) {
  return (
    <div className="space-y-4">
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-purple-900">
          🔄 Before & After Preprocessing
        </h3>
        <p className="text-xs text-purple-700 mt-1">
          See how preprocessing transforms each image. Left = original, Right =
          after normalization/resize.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {samples.map((sample: any, i: number) => (
          <div
            key={i}
            className="bg-white border border-gray-200 rounded-lg p-4"
          >
            <div className="text-xs font-medium text-gray-600 mb-2">
              Class: <span className="text-blue-600">{sample.class_name}</span>
            </div>
            <div className="flex items-center gap-4">
              {/* Before */}
              <div className="flex flex-col items-center">
                <PixelImage
                  pixels={sample.before_pixels}
                  width={sample.before_width}
                  height={sample.before_height}
                  size={80}
                  label="Before"
                />
                <div className="text-[9px] text-gray-400 mt-1">
                  [{sample.before_min}–{sample.before_max}]
                </div>
              </div>

              {/* Arrow */}
              <div className="text-gray-300 text-xl">→</div>

              {/* After */}
              <div className="flex flex-col items-center">
                <PixelImage
                  pixels={sample.after_pixels}
                  width={sample.after_width}
                  height={sample.after_height}
                  size={80}
                  label="After"
                />
                <div className="text-[9px] text-gray-400 mt-1">
                  [{sample.after_min}–{sample.after_max}]
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <p className="text-xs text-blue-700">
            Notice how pixel values change after normalization. The images may
            look {" "}
            <strong>visually similar</strong> because the brightness is
            relatively preserved, but the underlying numbers are now much
            smaller and more uniform — which is what the neural network needs.
          </p>
        </div>
      </div>
    </div>
  );
}

// --- Normalization Lab Tab ---

function NormLabTab({ data }: { data: any }) {
  const [selected, setSelected] = useState<string>("raw");
  const methods = data.methods;

  return (
    <div className="space-y-4">
      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-indigo-900">
          🧪 Normalization Lab
        </h3>
        <p className="text-xs text-indigo-700 mt-1">
          Compare how different normalization methods transform pixel values.
          Click each method to see its effect.
        </p>
      </div>

      {/* Method selector */}
      <div className="flex flex-wrap gap-2">
        {Object.entries(methods).map(([key, method]: [string, any]) => (
          <button
            key={key}
            onClick={() => setSelected(key)}
            className={`px-3 py-2 text-sm rounded-lg border transition-all ${
              selected === key
                ? "bg-blue-600 text-white border-blue-600 shadow-md"
                : "bg-white text-gray-700 border-gray-300 hover:border-blue-400"
            }`}
          >
            {method.label}
          </button>
        ))}
      </div>

      {/* Selected method details */}
      {methods[selected] && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-800 mb-1">
            {methods[selected].label}
          </h4>
          <p className="text-xs text-gray-600 mb-4">
            {methods[selected].description}
          </p>

          <div className="grid grid-cols-4 gap-3">
            <div className="bg-gray-50 rounded-lg p-3 text-center">
              <div className="text-[10px] text-gray-500 uppercase">Min</div>
              <div className="text-lg font-bold text-gray-900">
                {methods[selected].min}
              </div>
            </div>
            <div className="bg-gray-50 rounded-lg p-3 text-center">
              <div className="text-[10px] text-gray-500 uppercase">Max</div>
              <div className="text-lg font-bold text-gray-900">
                {methods[selected].max}
              </div>
            </div>
            <div className="bg-gray-50 rounded-lg p-3 text-center">
              <div className="text-[10px] text-gray-500 uppercase">Mean</div>
              <div className="text-lg font-bold text-blue-600">
                {methods[selected].mean}
              </div>
            </div>
            <div className="bg-gray-50 rounded-lg p-3 text-center">
              <div className="text-[10px] text-gray-500 uppercase">Std</div>
              <div className="text-lg font-bold text-purple-600">
                {methods[selected].std}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Comparison table */}
      <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="text-left px-3 py-2 text-xs text-gray-600">
                Method
              </th>
              <th className="text-center px-3 py-2 text-xs text-gray-600">
                Range
              </th>
              <th className="text-center px-3 py-2 text-xs text-gray-600">
                Mean
              </th>
              <th className="text-center px-3 py-2 text-xs text-gray-600">
                Std
              </th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(methods).map(([key, m]: [string, any]) => (
              <tr
                key={key}
                className={`border-t ${
                  selected === key ? "bg-blue-50" : "hover:bg-gray-50"
                }`}
              >
                <td className="px-3 py-2 font-medium">{m.label}</td>
                <td className="px-3 py-2 text-center font-mono text-xs">
                  [{m.min}, {m.max}]
                </td>
                <td className="px-3 py-2 text-center font-mono text-xs">
                  {m.mean}
                </td>
                <td className="px-3 py-2 text-center font-mono text-xs">
                  {m.std}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="bg-green-50 border border-green-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-green-600 mt-0.5" />
          <p className="text-xs text-green-700">
            <strong>Recommendation:</strong> {data.recommendation}
          </p>
        </div>
      </div>
    </div>
  );
}

// --- Histogram Tab ---

function HistogramTab({ data }: { data: any }) {
  const renderHistogram = (histData: any, color: string) => {
    const maxCount = Math.max(...histData.counts, 1);
    return (
      <div>
        <h4 className="text-xs font-semibold text-gray-700 mb-2">
          {histData.label}
        </h4>
        <div className="flex items-end gap-[2px] h-24">
          {histData.counts.map((count: number, i: number) => {
            const height = (count / maxCount) * 100;
            return (
              <div
                key={i}
                className={`flex-1 ${color} hover:opacity-80 transition-opacity rounded-t`}
                style={{
                  height: `${height}%`,
                  minHeight: count > 0 ? 2 : 0,
                }}
                title={`${histData.bin_edges[i]}–${histData.bin_edges[i + 1]}: ${count.toLocaleString()}`}
              />
            );
          })}
        </div>
        <div className="flex justify-between text-[10px] text-gray-500 mt-1">
          <span>{histData.min}</span>
          <span>{histData.max}</span>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      <div className="bg-orange-50 border border-orange-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-orange-900">
          📊 Pixel Value Histograms
        </h3>
        <p className="text-xs text-orange-700 mt-1">
          Compare pixel value distributions before and after{" "}
          {data.method?.replace("_", " ")} normalization.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          {renderHistogram(data.before, "bg-red-400")}
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          {renderHistogram(data.after, "bg-green-500")}
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <p className="text-xs text-blue-700">
            Notice how the distribution shape is preserved but the{" "}
            <strong>x-axis scale</strong> changes dramatically. This is exactly
            what normalization does — it keeps the relative relationships
            between pixels but scales them to a range that neural networks
            prefer.
          </p>
        </div>
      </div>
    </div>
  );
}
