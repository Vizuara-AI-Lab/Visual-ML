/**
 * ImageDatasetExplorer - Interactive learning activities for the Image Dataset node.
 * Tabs: Results | Sample Gallery | Class Distribution | Pixel Stats | Quiz
 */

import { useState, useMemo } from "react";
import {
  Image,
  BarChart3,
  HelpCircle,
  ClipboardList,
  Activity,
  CheckCircle,
  XCircle,
  Trophy,
  Info,
} from "lucide-react";

// --- Shared Image Renderer ---
function PixelImage({
  pixels,
  width,
  height,
  size = 64,
  label,
  highlight,
  onClick,
}: {
  pixels: number[];
  width: number;
  height: number;
  size?: number;
  label?: string;
  highlight?: boolean;
  onClick?: () => void;
}) {
  const canvasId = useMemo(() => `img-${Math.random().toString(36).slice(2)}`, []);

  // Render to canvas on mount
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
    <div
      className={`flex flex-col items-center gap-1 ${onClick ? "cursor-pointer hover:opacity-80" : ""}`}
      onClick={onClick}
    >
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          width: size,
          height: size,
          imageRendering: "pixelated",
          border: highlight ? "2px solid #3b82f6" : "1px solid #e5e7eb",
          borderRadius: 4,
        }}
      />
      {label && (
        <span className="text-[10px] text-gray-600 truncate max-w-[80px]">
          {label}
        </span>
      )}
    </div>
  );
}

// --- Main Explorer ---

interface ImageDatasetExplorerProps {
  result: any;
}

type ExplorerTab = "gallery" | "distribution" | "pixels" | "quiz";

export const ImageDatasetExplorer = ({
  result,
}: ImageDatasetExplorerProps) => {
  const [activeTab, setActiveTab] = useState<ExplorerTab>("gallery");

  const tabs: {
    id: ExplorerTab;
    label: string;
    icon: any;
    available: boolean;
  }[] = [
    {
      id: "gallery",
      label: "Sample Gallery",
      icon: Image,
      available: !!result.sample_images && result.sample_images.length > 0,
    },
    {
      id: "distribution",
      label: "Class Distribution",
      icon: BarChart3,
      available: !!result.class_distribution,
    },
    {
      id: "pixels",
      label: "Pixel Stats",
      icon: Activity,
      available: !!result.pixel_statistics,
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
                    ? "border-blue-600 text-blue-600"
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
      {activeTab === "gallery" && result.sample_images && (
        result.sample_images[0]?.data_type === "pose"
          ? <PoseGalleryTab data={result.sample_images} />
          : <GalleryTab data={result.sample_images} />
      )}
      {activeTab === "distribution" && result.class_distribution && (
        <DistributionTab data={result.class_distribution} />
      )}
      {activeTab === "pixels" && result.pixel_statistics && (
        <PixelStatsTab data={result.pixel_statistics} />
      )}
      {activeTab === "quiz" && result.quiz_questions && (
        <QuizTab questions={result.quiz_questions} />
      )}
    </div>
  );
};

// --- Gallery Tab ---

function GalleryTab({ data }: { data: any[] }) {
  const [selectedClass, setSelectedClass] = useState<number>(0);
  const [zoomedImage, setZoomedImage] = useState<any>(null);

  const classData = data[selectedClass];
  if (!classData) return null;

  return (
    <div className="space-y-4">
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-purple-900">
          📸 Sample Image Gallery
        </h3>
        <p className="text-xs text-purple-700 mt-1">
          Explore sample images from each class. Click an image to zoom in and
          see pixel values.
        </p>
      </div>

      {/* Class selector pills */}
      <div className="flex flex-wrap gap-2">
        {data.map((cls: any, i: number) => (
          <button
            key={i}
            onClick={() => setSelectedClass(i)}
            className={`px-3 py-1.5 text-sm rounded-lg border transition-all ${
              selectedClass === i
                ? "bg-blue-600 text-white border-blue-600 shadow-md"
                : "bg-white text-gray-700 border-gray-300 hover:border-blue-400"
            }`}
          >
            {cls.class_name}
            <span className="ml-1 text-xs opacity-75">({cls.count})</span>
          </button>
        ))}
      </div>

      {/* Image grid */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-semibold text-gray-800">
            {classData.class_name}
          </h4>
          <span className="text-xs text-gray-500">
            {classData.count} images total — showing {classData.samples.length}{" "}
            samples
          </span>
        </div>
        <div className="flex flex-wrap gap-3">
          {classData.samples.map((sample: any, i: number) => (
            <PixelImage
              key={i}
              pixels={sample.pixels}
              width={classData.width}
              height={classData.height}
              size={80}
              onClick={() => setZoomedImage(sample)}
              label={`μ=${sample.mean}`}
            />
          ))}
        </div>
      </div>

      {/* Zoomed image detail */}
      {zoomedImage && (
        <div className="bg-gray-50 border border-gray-300 rounded-lg p-4">
          <div className="flex items-start gap-6">
            <PixelImage
              pixels={zoomedImage.pixels}
              width={classData.width}
              height={classData.height}
              size={160}
              highlight
            />
            <div className="flex-1">
              <h4 className="text-sm font-semibold text-gray-800 mb-2">
                Image Details
              </h4>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-white rounded p-2 border">
                  <div className="text-[10px] text-gray-500 uppercase">Min</div>
                  <div className="text-sm font-bold text-gray-900">
                    {zoomedImage.min.toFixed(1)}
                  </div>
                </div>
                <div className="bg-white rounded p-2 border">
                  <div className="text-[10px] text-gray-500 uppercase">Max</div>
                  <div className="text-sm font-bold text-gray-900">
                    {zoomedImage.max.toFixed(1)}
                  </div>
                </div>
                <div className="bg-white rounded p-2 border">
                  <div className="text-[10px] text-gray-500 uppercase">Mean</div>
                  <div className="text-sm font-bold text-gray-900">
                    {zoomedImage.mean}
                  </div>
                </div>
                <div className="bg-white rounded p-2 border">
                  <div className="text-[10px] text-gray-500 uppercase">Pixels</div>
                  <div className="text-sm font-bold text-gray-900">
                    {classData.width}×{classData.height}
                  </div>
                </div>
              </div>
              <button
                onClick={() => setZoomedImage(null)}
                className="mt-3 text-xs text-blue-600 hover:underline"
              >
                Close zoom
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <p className="text-xs text-blue-700">
            Each image is stored as a row of pixel values. A{" "}
            <strong>{classData.width}×{classData.height}</strong> image has{" "}
            <strong>{classData.width * classData.height}</strong> pixel features.
            Click any image to see its pixel statistics.
          </p>
        </div>
      </div>
    </div>
  );
}

// --- Pose Gallery Tab (skeleton mini-diagrams instead of pixel images) ---

function PoseGalleryTab({ data }: { data: any[] }) {
  const [selectedClass, setSelectedClass] = useState<number>(0);
  const classData = data[selectedClass];
  if (!classData) return null;

  // Rough 2D positions for MediaPipe 33 landmarks
  const LANDMARK_POS: [number, number][] = [
    [0.5, 0.08], [0.47, 0.06], [0.45, 0.06], [0.43, 0.06],
    [0.53, 0.06], [0.55, 0.06], [0.57, 0.06], [0.38, 0.08],
    [0.62, 0.08], [0.48, 0.11], [0.52, 0.11], [0.35, 0.22],
    [0.65, 0.22], [0.28, 0.35], [0.72, 0.35], [0.22, 0.48],
    [0.78, 0.48], [0.20, 0.50], [0.80, 0.50], [0.21, 0.49],
    [0.79, 0.49], [0.23, 0.48], [0.77, 0.48], [0.40, 0.52],
    [0.60, 0.52], [0.38, 0.70], [0.62, 0.70], [0.37, 0.88],
    [0.63, 0.88], [0.35, 0.92], [0.65, 0.92], [0.37, 0.95],
    [0.63, 0.95],
  ];

  const CONNECTIONS = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    [11, 23], [12, 24], [23, 24],
    [23, 25], [25, 27], [24, 26], [26, 28],
    [27, 29], [27, 31], [28, 30], [28, 32],
    [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],
  ];

  const renderMiniSkeleton = (sample: any, size: number = 80) => {
    // Use actual landmark positions if available, otherwise fall back to default
    const landmarks = sample.landmarks || [];
    const useActual = landmarks.length === 33;

    return (
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="bg-slate-50 rounded border border-gray-200">
        {CONNECTIONS.map(([a, b], i) => {
          const ax = useActual ? landmarks[a].x * size : LANDMARK_POS[a][0] * size;
          const ay = useActual ? landmarks[a].y * size : LANDMARK_POS[a][1] * size;
          const bx = useActual ? landmarks[b].x * size : LANDMARK_POS[b][0] * size;
          const by = useActual ? landmarks[b].y * size : LANDMARK_POS[b][1] * size;
          return (
            <line key={i} x1={ax} y1={ay} x2={bx} y2={by} stroke="#94a3b8" strokeWidth={1.5} />
          );
        })}
        {(useActual ? landmarks : LANDMARK_POS.map(([x, y]: [number, number]) => ({ x, y }))).map((lm: any, i: number) => (
          <circle
            key={i}
            cx={lm.x * (useActual ? size : 1)}
            cy={lm.y * (useActual ? size : 1)}
            r={2}
            fill="#0ea5e9"
          />
        ))}
      </svg>
    );
  };

  return (
    <div className="space-y-4">
      <div className="bg-sky-50 border border-sky-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-sky-900">Pose Gallery</h3>
        <p className="text-xs text-sky-700 mt-1">
          Sample skeleton poses from each class. Each pose has {classData.n_landmarks || 33} body landmarks.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {data.map((cls: any, i: number) => (
          <button
            key={i}
            onClick={() => setSelectedClass(i)}
            className={`px-3 py-1.5 text-sm rounded-lg border transition-all ${
              selectedClass === i
                ? "bg-sky-600 text-white border-sky-600 shadow-md"
                : "bg-white text-gray-700 border-gray-300 hover:border-sky-400"
            }`}
          >
            {cls.class_name}
            <span className="ml-1 text-xs opacity-75">({cls.count})</span>
          </button>
        ))}
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-semibold text-gray-800">{classData.class_name}</h4>
          <span className="text-xs text-gray-500">
            {classData.count} samples total — showing {classData.samples.length}
          </span>
        </div>
        <div className="flex flex-wrap gap-3">
          {classData.samples.map((sample: any, i: number) => (
            <div key={i} className="flex flex-col items-center gap-1">
              {renderMiniSkeleton(sample)}
              <span className="text-[10px] text-gray-500">Sample {i + 1}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <p className="text-xs text-blue-700">
            Each pose is stored as <strong>132 numbers</strong> (33 landmarks × 4 values: x, y, z, visibility).
            Unlike pixel images, pose data captures body geometry directly, making it invariant to background and lighting.
          </p>
        </div>
      </div>
    </div>
  );
}

// --- Distribution Tab ---

function DistributionTab({ data }: { data: any }) {
  const maxCount = Math.max(...data.classes.map((c: any) => c.count), 1);

  return (
    <div className="space-y-4">
      <div className="bg-green-50 border border-green-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-green-900">
          📊 Class Distribution
        </h3>
        <p className="text-xs text-green-700 mt-1">
          {data.total_samples} images across {data.n_classes} classes.{" "}
          {data.balance_message}
        </p>
      </div>

      {/* Bar chart */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="space-y-3">
          {data.classes.map((cls: any, i: number) => {
            const barWidth = (cls.count / maxCount) * 100;
            const colors = [
              "bg-blue-500",
              "bg-emerald-500",
              "bg-purple-500",
              "bg-orange-500",
              "bg-pink-500",
              "bg-cyan-500",
              "bg-yellow-500",
              "bg-red-500",
              "bg-indigo-500",
              "bg-teal-500",
            ];
            return (
              <div key={i} className="flex items-center gap-3">
                <div className="w-24 text-right">
                  <span className="text-sm font-medium text-gray-800 truncate">
                    {cls.name}
                  </span>
                </div>
                <div className="flex-1 bg-gray-100 rounded-full h-6 overflow-hidden">
                  <div
                    className={`h-full ${colors[i % colors.length]} rounded-full flex items-center pl-2 transition-all duration-500`}
                    style={{ width: `${barWidth}%` }}
                  >
                    {barWidth > 15 && (
                      <span className="text-[10px] text-white font-bold">
                        {cls.count}
                      </span>
                    )}
                  </div>
                </div>
                <div className="w-16 text-right text-xs text-gray-500">
                  {cls.percentage}%
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Balance indicator */}
      <div
        className={`rounded-lg p-3 border ${
          data.is_balanced
            ? "bg-green-50 border-green-300"
            : "bg-orange-50 border-orange-300"
        }`}
      >
        <div className="flex items-center gap-2">
          {data.is_balanced ? (
            <CheckCircle className="w-5 h-5 text-green-600" />
          ) : (
            <Info className="w-5 h-5 text-orange-600" />
          )}
          <div>
            <div
              className={`text-sm font-semibold ${
                data.is_balanced ? "text-green-800" : "text-orange-800"
              }`}
            >
              {data.is_balanced ? "✅ Balanced" : "⚠️ Imbalanced"} (Ratio:{" "}
              {data.imbalance_ratio}:1)
            </div>
            <p
              className={`text-xs mt-0.5 ${
                data.is_balanced ? "text-green-700" : "text-orange-700"
              }`}
            >
              {data.balance_message}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// --- Pixel Stats Tab ---

function PixelStatsTab({ data }: { data: any }) {
  const histogram = data.histogram;
  const maxCount = Math.max(...histogram.counts, 1);

  return (
    <div className="space-y-4">
      <div className="bg-orange-50 border border-orange-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-orange-900">
          📈 Pixel Value Statistics
        </h3>
        <p className="text-xs text-orange-700 mt-1">
          Distribution of pixel values across all images in the dataset.
        </p>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-xs text-gray-500 uppercase">Global Min</div>
          <div className="text-lg font-bold text-gray-900">
            {data.global_min.toFixed(1)}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-xs text-gray-500 uppercase">Global Max</div>
          <div className="text-lg font-bold text-gray-900">
            {data.global_max.toFixed(1)}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-xs text-gray-500 uppercase">Mean</div>
          <div className="text-lg font-bold text-blue-600">
            {data.global_mean}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-xs text-gray-500 uppercase">Std Dev</div>
          <div className="text-lg font-bold text-purple-600">
            {data.global_std}
          </div>
        </div>
      </div>

      {/* Histogram */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-800 mb-3">
          Pixel Value Histogram
        </h4>
        <div className="flex items-end gap-[2px] h-32">
          {histogram.counts.map((count: number, i: number) => {
            const height = (count / maxCount) * 100;
            return (
              <div
                key={i}
                className="flex-1 bg-blue-400 hover:bg-blue-600 transition-colors rounded-t"
                style={{ height: `${height}%`, minHeight: count > 0 ? 2 : 0 }}
                title={`${histogram.bin_edges[i].toFixed(1)}–${histogram.bin_edges[i + 1]?.toFixed(1)}: ${count.toLocaleString()} pixels`}
              />
            );
          })}
        </div>
        <div className="flex justify-between text-[10px] text-gray-500 mt-1">
          <span>{histogram.bin_edges[0]}</span>
          <span>Pixel Value →</span>
          <span>{histogram.bin_edges[histogram.bin_edges.length - 1]}</span>
        </div>
      </div>

      {/* Insights */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
          <div className="text-xs text-gray-500">Zero Pixels (Black)</div>
          <div className="text-lg font-bold text-gray-900">
            {data.zero_pixel_pct}%
          </div>
          <div className="text-[10px] text-gray-500">
            High % = dark backgrounds
          </div>
        </div>
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
          <div className="text-xs text-gray-500">Non-zero Mean</div>
          <div className="text-lg font-bold text-gray-900">
            {data.nonzero_mean}
          </div>
          <div className="text-[10px] text-gray-500">
            Average brightness of foreground
          </div>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <p className="text-xs text-blue-700">
            <strong>Why pixel stats matter:</strong> The pixel value range tells
            you whether normalization is needed. If values range 0-255, you'll
            want to normalize them to 0-1 before training. The histogram shape
            reveals if the data is bimodal (foreground vs background) which is
            common in digit/shape datasets.
          </p>
        </div>
      </div>
    </div>
  );
}

// --- Quiz Tab (reusable) ---

export function QuizTab({ questions }: { questions: any[] }) {
  const [currentQ, setCurrentQ] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [answered, setAnswered] = useState(false);
  const [score, setScore] = useState(0);
  const [answers, setAnswers] = useState<(number | null)[]>(
    new Array(questions.length).fill(null)
  );
  const [showResults, setShowResults] = useState(false);

  const question = questions[currentQ];

  const handleSelect = (optionIdx: number) => {
    if (answered) return;
    setSelectedAnswer(optionIdx);
    setAnswered(true);
    const newAnswers = [...answers];
    newAnswers[currentQ] = optionIdx;
    setAnswers(newAnswers);
    if (optionIdx === question.correct_answer) setScore((s) => s + 1);
  };

  const handleNext = () => {
    if (currentQ < questions.length - 1) {
      setCurrentQ((q) => q + 1);
      setSelectedAnswer(null);
      setAnswered(false);
    } else {
      setShowResults(true);
    }
  };

  const handleRetry = () => {
    setCurrentQ(0);
    setSelectedAnswer(null);
    setAnswered(false);
    setScore(0);
    setAnswers(new Array(questions.length).fill(null));
    setShowResults(false);
  };

  if (showResults) {
    const percentage = Math.round((score / questions.length) * 100);
    return (
      <div className="space-y-4">
        <div
          className={`text-center py-8 rounded-lg border-2 ${
            percentage >= 80
              ? "bg-green-50 border-green-300"
              : percentage >= 50
                ? "bg-yellow-50 border-yellow-300"
                : "bg-red-50 border-red-300"
          }`}
        >
          <Trophy
            className={`w-12 h-12 mx-auto mb-3 ${
              percentage >= 80
                ? "text-green-600"
                : percentage >= 50
                  ? "text-yellow-600"
                  : "text-red-600"
            }`}
          />
          <div className="text-3xl font-bold text-gray-900 mb-2">
            {score}/{questions.length}
          </div>
          <p className="text-sm text-gray-700">
            {percentage >= 80
              ? "Excellent understanding!"
              : percentage >= 50
                ? "Good effort! Review the explanations."
                : "Keep learning! Try again after reviewing."}
          </p>
        </div>
        <button
          onClick={handleRetry}
          className="w-full py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium"
        >
          Try Again
        </button>
        <div className="space-y-3">
          {questions.map((q: any, qi: number) => (
            <div
              key={qi}
              className="bg-gray-50 border border-gray-200 rounded-lg p-3 text-sm"
            >
              <p className="font-medium text-gray-800 mb-1">{q.question}</p>
              <p
                className={`text-xs ${
                  answers[qi] === q.correct_answer
                    ? "text-green-700"
                    : "text-red-700"
                }`}
              >
                Your answer: {q.options[answers[qi] ?? 0]}{" "}
                {answers[qi] === q.correct_answer ? "✓" : "✗"}
              </p>
              <p className="text-xs text-blue-700 mt-1">{q.explanation}</p>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-indigo-900">
          🧠 Knowledge Check
        </h3>
        <p className="text-xs text-indigo-700 mt-1">
          Test your understanding with questions generated from YOUR data.
        </p>
      </div>

      {/* Progress dots */}
      <div className="flex justify-center gap-2">
        {questions.map((_: any, i: number) => (
          <div
            key={i}
            className={`w-3 h-3 rounded-full transition-all ${
              i === currentQ
                ? "bg-blue-600 scale-125"
                : i < currentQ
                  ? answers[i] === questions[i].correct_answer
                    ? "bg-green-500"
                    : "bg-red-400"
                  : "bg-gray-300"
            }`}
          />
        ))}
      </div>

      {/* Question card */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <div className="text-xs text-gray-500 mb-2">
          Question {currentQ + 1} of {questions.length}
          {question.difficulty && (
            <span
              className={`ml-2 px-2 py-0.5 rounded-full text-[10px] font-medium ${
                question.difficulty === "easy"
                  ? "bg-green-100 text-green-700"
                  : question.difficulty === "medium"
                    ? "bg-yellow-100 text-yellow-700"
                    : "bg-red-100 text-red-700"
              }`}
            >
              {question.difficulty}
            </span>
          )}
        </div>
        <p className="text-sm font-medium text-gray-900 mb-4">
          {question.question}
        </p>

        <div className="space-y-2">
          {question.options.map((option: string, oi: number) => {
            let optionClass =
              "bg-white border-gray-300 hover:border-blue-400 cursor-pointer";
            if (answered) {
              if (oi === question.correct_answer) {
                optionClass = "bg-green-50 border-green-500 text-green-900";
              } else if (oi === selectedAnswer) {
                optionClass = "bg-red-50 border-red-500 text-red-900";
              } else {
                optionClass = "bg-gray-50 border-gray-200 text-gray-400";
              }
            }
            return (
              <button
                key={oi}
                onClick={() => handleSelect(oi)}
                className={`w-full text-left px-4 py-3 border rounded-lg text-sm transition-all ${optionClass}`}
              >
                <div className="flex items-center gap-3">
                  <span className="w-5 h-5 rounded-full border-2 flex items-center justify-center text-xs shrink-0">
                    {answered &&
                      oi === question.correct_answer && (
                        <CheckCircle className="w-4 h-4 text-green-600" />
                      )}
                    {answered &&
                      oi === selectedAnswer &&
                      oi !== question.correct_answer && (
                        <XCircle className="w-4 h-4 text-red-600" />
                      )}
                    {!answered && String.fromCharCode(65 + oi)}
                  </span>
                  {option}
                </div>
              </button>
            );
          })}
        </div>

        {answered && (
          <div className="mt-4 bg-blue-50 border border-blue-200 rounded-lg p-3">
            <p className="text-xs text-blue-800">{question.explanation}</p>
          </div>
        )}

        {answered && (
          <button
            onClick={handleNext}
            className="mt-3 w-full py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium"
          >
            {currentQ < questions.length - 1 ? "Next Question →" : "See Results"}
          </button>
        )}
      </div>
    </div>
  );
}
