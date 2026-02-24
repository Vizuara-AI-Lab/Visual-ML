/**
 * ImageAugmentationExplorer - Interactive learning for Data Augmentation node.
 * Tabs: Results | Augmentation Gallery | Transform Lab | Dataset Growth | Quiz
 */

import { useState, type ReactNode } from "react";
import {
  ClipboardList,
  Image,
  FlaskConical,
  TrendingUp,
  HelpCircle,
  Info,
} from "lucide-react";
import { QuizTab } from "./ImageDatasetExplorer";

// --- Shared image renderer ---
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

interface ImageAugmentationExplorerProps {
  result: any;
  renderResults: () => ReactNode;
}

type Tab = "results" | "gallery" | "transforms" | "growth" | "quiz";

export const ImageAugmentationExplorer = ({
  result,
  renderResults,
}: ImageAugmentationExplorerProps) => {
  const [activeTab, setActiveTab] = useState<Tab>("results");

  const tabs: { id: Tab; label: string; icon: any; available: boolean }[] = [
    { id: "results", label: "Results", icon: ClipboardList, available: true },
    {
      id: "gallery",
      label: "Augmentation Gallery",
      icon: Image,
      available: !!result.augmentation_gallery && result.augmentation_gallery.length > 0,
    },
    {
      id: "transforms",
      label: "Transform Lab",
      icon: FlaskConical,
      available: !!result.transform_effects,
    },
    {
      id: "growth",
      label: "Dataset Growth",
      icon: TrendingUp,
      available: !!result.dataset_growth,
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
      {activeTab === "gallery" && result.augmentation_gallery && (
        <AugGalleryTab data={result.augmentation_gallery} />
      )}
      {activeTab === "transforms" && result.transform_effects && (
        <TransformLabTab data={result.transform_effects} />
      )}
      {activeTab === "growth" && result.dataset_growth && (
        <DatasetGrowthTab data={result.dataset_growth} />
      )}
      {activeTab === "quiz" && result.quiz_questions && (
        <QuizTab questions={result.quiz_questions} />
      )}
    </div>
  );
};

// --- Augmentation Gallery Tab ---

function AugGalleryTab({ data }: { data: any[] }) {
  return (
    <div className="space-y-4">
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-purple-900">
          🎨 Augmentation Gallery
        </h3>
        <p className="text-xs text-purple-700 mt-1">
          See how augmentation creates new training variations. Left = original,
          Right = augmented version.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {data.map((pair: any, i: number) => (
          <div
            key={i}
            className="bg-white border border-gray-200 rounded-lg p-4"
          >
            <div className="text-xs font-medium text-gray-600 mb-2">
              Class: <span className="text-blue-600">{pair.class_name}</span>
            </div>
            <div className="flex items-center gap-4 justify-center">
              <PixelImage
                pixels={pair.original_pixels}
                width={pair.width}
                height={pair.height}
                size={72}
                label="Original"
              />
              <div className="text-gray-300 text-xl">→</div>
              <PixelImage
                pixels={pair.augmented_pixels}
                width={pair.width}
                height={pair.height}
                size={72}
                label="Augmented"
              />
            </div>
          </div>
        ))}
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <p className="text-xs text-blue-700">
            Each augmented image is a unique variation — flipped, rotated,
            brightened, or with added noise. The model sees each variation as a{" "}
            <strong>new training example</strong>, learning to recognize the
            same class despite visual changes.
          </p>
        </div>
      </div>
    </div>
  );
}

// --- Transform Lab Tab ---

function TransformLabTab({ data }: { data: any }) {
  const [selectedEffect, setSelectedEffect] = useState<string>("original");
  const effects = data.effects || {};

  return (
    <div className="space-y-4">
      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-indigo-900">
          🔬 Transform Lab
        </h3>
        <p className="text-xs text-indigo-700 mt-1">
          See each augmentation transform applied individually. Click a
          transform to see its effect on a sample image.
        </p>
      </div>

      {/* Transform selector */}
      <div className="flex flex-wrap gap-2">
        {Object.entries(effects).map(([key, effect]: [string, any]) => (
          <button
            key={key}
            onClick={() => setSelectedEffect(key)}
            className={`px-3 py-2 text-sm rounded-lg border transition-all ${
              selectedEffect === key
                ? "bg-blue-600 text-white border-blue-600 shadow-md"
                : "bg-white text-gray-700 border-gray-300 hover:border-blue-400"
            }`}
          >
            {effect.label}
          </button>
        ))}
      </div>

      {/* Side-by-side: Original vs Selected */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <div className="flex items-center justify-center gap-8">
          <div className="text-center">
            <PixelImage
              pixels={effects.original?.pixels || []}
              width={data.width}
              height={data.height}
              size={120}
            />
            <div className="text-xs font-medium text-gray-600 mt-2">
              Original
            </div>
          </div>

          {selectedEffect !== "original" && (
            <>
              <div className="text-3xl text-gray-300">→</div>
              <div className="text-center">
                <PixelImage
                  pixels={effects[selectedEffect]?.pixels || []}
                  width={data.width}
                  height={data.height}
                  size={120}
                />
                <div className="text-xs font-medium text-blue-600 mt-2">
                  {effects[selectedEffect]?.label}
                </div>
              </div>
            </>
          )}
        </div>

        {effects[selectedEffect] && (
          <div className="mt-4 bg-gray-50 rounded-lg p-3">
            <p className="text-xs text-gray-700">
              {effects[selectedEffect].description}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

// --- Dataset Growth Tab ---

function DatasetGrowthTab({ data }: { data: any }) {
  const maxCount = Math.max(
    ...data.per_class.map((c: any) => c.total),
    1
  );

  return (
    <div className="space-y-4">
      <div className="bg-green-50 border border-green-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-green-900">
          📈 Dataset Growth
        </h3>
        <p className="text-xs text-green-700 mt-1">
          Augmentation expanded the dataset from {data.original_total} to{" "}
          {data.final_total} images ({data.growth_factor}× growth).
        </p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-3 gap-3">
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-xs text-gray-500">Original</div>
          <div className="text-2xl font-bold text-gray-900">
            {data.original_total.toLocaleString()}
          </div>
        </div>
        <div className="bg-white border border-blue-200 rounded-lg p-3 text-center">
          <div className="text-xs text-blue-500">+ Augmented</div>
          <div className="text-2xl font-bold text-blue-600">
            {data.augmented_added.toLocaleString()}
          </div>
        </div>
        <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3 text-center">
          <div className="text-xs text-emerald-500">Total</div>
          <div className="text-2xl font-bold text-emerald-600">
            {data.final_total.toLocaleString()}
          </div>
        </div>
      </div>

      {/* Per-class breakdown */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-800 mb-3">
          Per-Class Growth
        </h4>
        <div className="space-y-3">
          {data.per_class.map((cls: any, i: number) => {
            const origWidth = (cls.original / maxCount) * 100;
            const addedWidth = (cls.added / maxCount) * 100;
            return (
              <div key={i} className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="font-medium text-gray-700">
                    {cls.class_name}
                  </span>
                  <span className="text-gray-500">
                    {cls.original} → {cls.total}
                  </span>
                </div>
                <div className="flex h-4 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className="bg-gray-400 h-full"
                    style={{ width: `${origWidth}%` }}
                    title={`Original: ${cls.original}`}
                  />
                  <div
                    className="bg-blue-400 h-full"
                    style={{ width: `${addedWidth}%` }}
                    title={`Added: ${cls.added}`}
                  />
                </div>
              </div>
            );
          })}
        </div>
        <div className="flex items-center gap-4 mt-3 text-xs text-gray-500">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 bg-gray-400 rounded-sm" /> Original
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 bg-blue-400 rounded-sm" /> Augmented
          </span>
        </div>
      </div>
    </div>
  );
}
