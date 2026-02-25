/**
 * ImagePredictionsExplorer - Interactive learning for Image Predictions node.
 * Tabs: Results | Prediction Gallery | Confusion Analysis | Confidence | Quiz
 *       | Live Camera | Draw | Upload Test
 */

import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import {
  Activity,
  Image as ImageIcon,
  BarChart3,
  HelpCircle,
  Info,
  CheckCircle,
  XCircle,
  Video,
  PenTool,
  Upload,
  Layers,
  ChevronRight,
} from "lucide-react";
import { QuizTab } from "./ImageDatasetExplorer";
import { apiClient } from "../../lib/axios";
import { LiveCameraTestPanel } from "./LiveCameraTestPanel";
import { DrawingCanvas } from "./DrawingCanvas";
import { ImageUploadTest } from "./ImageUploadTest";

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
}

type Tab = "overview" | "architecture" | "gallery" | "confidence" | "quiz" | "live_camera" | "draw" | "upload_test";

export const ImagePredictionsExplorer = ({
  result,
}: ImagePredictionsExplorerProps) => {
  const [activeTab, setActiveTab] = useState<Tab>("overview");

  // Extract model info for testing tabs
  const modelPath: string = result.model_path || "";
  const imageWidth: number = result.image_width || result.prediction_gallery?.[0]?.width || 28;
  const imageHeight: number = result.image_height || result.prediction_gallery?.[0]?.height || 28;
  const classNames: string[] = result.class_names || [];
  const hasModel = !!modelPath;

  // Shared predict callback for all testing tabs
  const predictPixels = useCallback(
    async (pixels: number[]) => {
      const response = await apiClient.post("/ml/camera/predict", {
        model_path: modelPath,
        pixels,
      });
      const data = response.data as {
        class_name: string;
        confidence: number;
        all_scores: { class_name: string; score: number }[];
      };
      // Map numeric class indices to human-readable names
      const mapIdx = (raw: string) => {
        const idx = parseInt(raw);
        return !isNaN(idx) && classNames[idx] ? classNames[idx] : raw;
      };
      return {
        class_name: mapIdx(data.class_name),
        confidence: data.confidence,
        all_scores: data.all_scores.map((s) => ({
          class_name: mapIdx(s.class_name),
          score: s.score,
        })),
      };
    },
    [modelPath, classNames],
  );

  const tabs: { id: Tab; label: string; icon: any; available: boolean }[] = [
    { id: "overview", label: "Detection Overview", icon: Activity, available: true },
    {
      id: "architecture",
      label: "Architecture",
      icon: Layers,
      available: !!result.architecture_diagram,
    },
    {
      id: "gallery",
      label: "Prediction Gallery",
      icon: ImageIcon,
      available: !!result.prediction_gallery && result.prediction_gallery.length > 0,
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
    // --- Testing tabs (only when model is available) ---
    {
      id: "live_camera",
      label: "Live Camera",
      icon: Video,
      available: hasModel,
    },
    {
      id: "draw",
      label: "Draw",
      icon: PenTool,
      available: hasModel,
    },
    {
      id: "upload_test",
      label: "Test Image",
      icon: Upload,
      available: hasModel,
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

      {activeTab === "overview" && <DetectionOverviewTab result={result} />}
      {activeTab === "architecture" && result.architecture_diagram && (
        <ArchitectureTab data={result.architecture_diagram} />
      )}
      {activeTab === "gallery" && result.prediction_gallery && (
        <PredictionGalleryTab
          data={result.prediction_gallery}
          metrics={result.metrics_summary}
        />
      )}
      {activeTab === "confidence" && result.confidence_distribution && (
        <ConfidenceTab data={result.confidence_distribution} />
      )}
      {activeTab === "quiz" && result.quiz_questions && (
        <QuizTab questions={result.quiz_questions} />
      )}

      {/* --- Testing tabs --- */}
      {activeTab === "live_camera" && hasModel && (
        <LiveCameraTestPanel
          modelId={result.model_id || ""}
          modelPath={modelPath}
          classNames={classNames}
          imageWidth={imageWidth}
          imageHeight={imageHeight}
          onPredict={predictPixels}
        />
      )}
      {activeTab === "draw" && hasModel && (
        <DrawingCanvas
          imageWidth={imageWidth}
          imageHeight={imageHeight}
          classNames={classNames}
          onPredict={predictPixels}
        />
      )}
      {activeTab === "upload_test" && hasModel && (
        <ImageUploadTest
          imageWidth={imageWidth}
          imageHeight={imageHeight}
          classNames={classNames}
          onPredict={predictPixels}
        />
      )}
    </div>
  );
};

// --- Detection Overview Tab (animated) ---

function DetectionOverviewTab({ result }: { result: any }) {
  const accuracy = result.overall_accuracy ?? 0;
  const grade = result.metrics_summary?.grade ?? "?";
  const macroF1 = result.metrics_summary?.macro_f1 ?? 0;
  const trainSamples = result.training_samples ?? 0;
  const testSamples = result.test_samples ?? 0;
  const trainingTime = result.training_time_seconds ?? 0;
  const perClass = result.per_class_metrics ?? {};
  const classNames = result.class_names ?? [];

  // SVG gauge constants
  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  const gaugeColor =
    accuracy >= 0.8 ? "#22c55e" : accuracy >= 0.6 ? "#eab308" : "#ef4444";
  const gradeColor =
    grade === "A+" || grade === "A"
      ? "text-green-600"
      : grade === "B"
        ? "text-blue-600"
        : grade === "C"
          ? "text-yellow-600"
          : "text-red-600";

  const stats = [
    { label: "Training Samples", value: trainSamples.toLocaleString(), color: "bg-blue-50 border-blue-200", textColor: "text-blue-700" },
    { label: "Test Samples", value: testSamples.toLocaleString(), color: "bg-purple-50 border-purple-200", textColor: "text-purple-700" },
    { label: "Training Time", value: `${trainingTime.toFixed(1)}s`, color: "bg-amber-50 border-amber-200", textColor: "text-amber-700" },
    { label: "Macro F1", value: `${(macroF1 * 100).toFixed(1)}%`, color: "bg-emerald-50 border-emerald-200", textColor: "text-emerald-700" },
  ];

  return (
    <div className="space-y-6">
      {/* Section A: Accuracy Gauge */}
      <div className="flex flex-col items-center py-4">
        <div className="relative w-44 h-44">
          <svg width="176" height="176" viewBox="0 0 176 176">
            <circle
              cx="88" cy="88" r={radius}
              fill="none" stroke="#e5e7eb" strokeWidth="10"
            />
            <motion.circle
              cx="88" cy="88" r={radius}
              fill="none"
              stroke={gaugeColor}
              strokeWidth="10"
              strokeLinecap="round"
              strokeDasharray={circumference}
              initial={{ strokeDashoffset: circumference }}
              animate={{ strokeDashoffset: circumference * (1 - accuracy) }}
              transition={{ duration: 1.2, delay: 0.3, ease: "easeOut" }}
              transform="rotate(-90 88 88)"
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <motion.span
              className="text-3xl font-bold text-gray-900"
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.8 }}
            >
              {(accuracy * 100).toFixed(1)}%
            </motion.span>
            <motion.span
              className={`text-lg font-bold ${gradeColor}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1.2 }}
            >
              Grade: {grade}
            </motion.span>
          </div>
        </div>
        <motion.p
          className="text-sm text-gray-500 mt-2"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.0 }}
        >
          Test Accuracy
        </motion.p>
      </div>

      {/* Section B: Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {stats.map((stat, i) => (
          <motion.div
            key={stat.label}
            className={`border rounded-lg p-3 text-center ${stat.color}`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.2 + i * 0.1 }}
          >
            <div className="text-[11px] text-gray-500">{stat.label}</div>
            <div className={`text-xl font-bold ${stat.textColor}`}>{stat.value}</div>
          </motion.div>
        ))}
      </div>

      {/* Section C: Per-Class Score Bars */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-800 mb-3">Per-Class Performance (F1 Score)</h4>
        <div className="space-y-3">
          {classNames.map((name: string, i: number) => {
            const cls = perClass[name];
            if (!cls) return null;
            const f1 = cls.f1 ?? 0;
            const barColor =
              f1 >= 0.8 ? "bg-green-500" : f1 >= 0.6 ? "bg-yellow-500" : "bg-red-500";

            return (
              <div key={name}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-gray-700">{name}</span>
                  <span className="text-xs text-gray-500">
                    P: {(cls.precision * 100).toFixed(0)}% &middot; R: {(cls.recall * 100).toFixed(0)}% &middot; F1: {(f1 * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="w-full bg-gray-100 rounded-full h-3">
                  <motion.div
                    className={`h-full rounded-full ${barColor}`}
                    initial={{ width: 0 }}
                    animate={{ width: `${f1 * 100}%` }}
                    transition={{ duration: 0.8, delay: 0.5 + i * 0.15, ease: "easeOut" }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Section D: Full Training Pipeline Animation */}
      <TrainingPipelineAnimation result={result} />
    </div>
  );
}

// --- Training Pipeline Animation ---

function TrainingPipelineAnimation({ result }: { result: any }) {
  const w = result.image_width || 28;
  const h = result.image_height || 28;
  const nPixels = w * h;
  const layers: any[] = result.architecture_diagram?.layers || [];
  const lossCurve: number[] = result.loss_curve || [];
  const classNames: string[] = result.class_names || [];
  const perClass = result.per_class_metrics || {};
  const trainSamples = result.training_samples || 0;
  const activation = layers.find((l: any) => l.type === "hidden")?.activation || "relu";

  // --- Network SVG layout ---
  const NET_W = 460;
  const NET_H = 160;
  const maxShow = 7;

  const layerX = layers.map((_: any, i: number) =>
    50 + (i / Math.max(1, layers.length - 1)) * (NET_W - 100)
  );

  const buildNeurons = (size: number, x: number) => {
    const n = Math.min(size, maxShow);
    const gap = (NET_H - 40) / (n + 1);
    return Array.from({ length: n }, (_, j) => ({
      x,
      y: 20 + gap * (j + 1),
      ellipsis: size > maxShow && j === Math.floor(n / 2),
    }));
  };

  const layerNeurons = layers.map((l: any, i: number) => buildNeurons(l.size, layerX[i]));

  // --- Loss curve SVG path ---
  const lossPath = (() => {
    if (lossCurve.length < 2) return "";
    const maxL = Math.max(...lossCurve);
    const minL = Math.min(...lossCurve);
    const range = maxL - minL || 1;
    return lossCurve
      .map((v, i) => {
        const x = (i / (lossCurve.length - 1)) * 300;
        const y = 80 - ((v - minL) / range) * 70;
        return `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");
  })();

  // Deterministic pixel grid brightness values
  const pixelBrightness = Array.from({ length: 25 }, (_, i) => 0.15 + ((i * 7 + 3) % 10) / 12);

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 space-y-5">
      <h4 className="text-sm font-semibold text-gray-800">
        Training Pipeline — How Your Model Was Built
      </h4>

      {/* Step 1: Feature Extraction — Image → Pixel Vector */}
      <motion.div
        className="flex items-start gap-3"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div className="flex-shrink-0 w-7 h-7 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 text-xs font-bold">
          1
        </div>
        <div className="flex-1">
          <div className="text-xs font-medium text-gray-700 mb-1.5">Feature Extraction</div>
          <div className="flex items-center gap-3 flex-wrap">
            {/* Mini pixel grid */}
            <div className="grid grid-cols-5 gap-px" style={{ width: 40, height: 40 }}>
              {pixelBrightness.map((b, i) => (
                <motion.div
                  key={i}
                  className="rounded-[1px]"
                  style={{ backgroundColor: `rgba(55,65,81,${b})` }}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.3 + i * 0.015 }}
                />
              ))}
            </div>
            <ChevronRight className="w-4 h-4 text-gray-300 flex-shrink-0" />
            {/* Flattened pixel strip */}
            <div className="flex gap-px items-center">
              {Array.from({ length: 14 }, (_, i) => (
                <motion.div
                  key={i}
                  className="w-1.5 rounded-[1px]"
                  style={{
                    height: 18,
                    backgroundColor: `hsl(220, 50%, ${35 + i * 3}%)`,
                  }}
                  initial={{ scaleY: 0 }}
                  animate={{ scaleY: 1 }}
                  transition={{ delay: 0.5 + i * 0.025 }}
                />
              ))}
              <span className="text-[9px] text-gray-400 ml-0.5">...</span>
            </div>
            <div className="text-[10px] text-gray-500 ml-1">
              {w}x{h} image → <strong>{nPixels.toLocaleString()}</strong> pixel values
            </div>
          </div>
        </div>
      </motion.div>

      {/* Step 2: Normalization — [0-255] → [0-1] */}
      <motion.div
        className="flex items-start gap-3"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.6 }}
      >
        <div className="flex-shrink-0 w-7 h-7 rounded-full bg-cyan-100 flex items-center justify-center text-cyan-600 text-xs font-bold">
          2
        </div>
        <div className="flex-1">
          <div className="text-xs font-medium text-gray-700 mb-1.5">Pixel Normalization</div>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="px-2 py-0.5 bg-gray-100 rounded text-[11px] font-mono text-gray-600">
              [0 – 255]
            </span>
            <motion.span
              className="text-sm font-bold text-cyan-600"
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.9, type: "spring" }}
            >
              ÷ 255
            </motion.span>
            <ChevronRight className="w-3 h-3 text-gray-300" />
            <motion.span
              className="px-2 py-0.5 bg-cyan-50 border border-cyan-200 rounded text-[11px] font-mono text-cyan-700"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1.05 }}
            >
              [0.0 – 1.0]
            </motion.span>
            <span className="text-[10px] text-gray-500 ml-1">
              Same scale as live camera input
            </span>
          </div>
        </div>
      </motion.div>

      {/* Step 3: Data Augmentation */}
      <motion.div
        className="flex items-start gap-3"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 1.0 }}
      >
        <div className="flex-shrink-0 w-7 h-7 rounded-full bg-amber-100 flex items-center justify-center text-amber-600 text-xs font-bold">
          3
        </div>
        <div className="flex-1">
          <div className="text-xs font-medium text-gray-700 mb-1.5">Data Augmentation</div>
          <div className="flex items-center gap-2 flex-wrap">
            {/* Original image block */}
            <div className="w-8 h-8 bg-gray-300 rounded border border-gray-400 flex items-center justify-center text-[8px] text-gray-600">
              orig
            </div>
            <ChevronRight className="w-3 h-3 text-gray-300" />
            {/* Augmented copies with transformations */}
            {[
              { label: "shift", transform: "translate(2px,-1px)" },
              { label: "noise", transform: "rotate(-2deg)" },
              { label: "bright", transform: "translate(-1px,2px)" },
              { label: "shift", transform: "rotate(2deg) scale(0.97)" },
            ].map((aug, i) => (
              <motion.div
                key={i}
                className="w-7 h-7 bg-amber-100 border border-amber-300 rounded flex items-center justify-center text-[7px] text-amber-600"
                style={{ transform: aug.transform }}
                initial={{ opacity: 0, scale: 0 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 1.2 + i * 0.1, type: "spring" }}
              >
                {aug.label}
              </motion.div>
            ))}
            <div className="text-[10px] text-gray-500 ml-1">
              <strong>{trainSamples.toLocaleString()}</strong> images — shift ±2px, noise, brightness
            </div>
          </div>
        </div>
      </motion.div>

      {/* Step 4: MLP Neural Network — Forward Pass */}
      {layers.length > 0 && (
        <motion.div
          className="flex items-start gap-3"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.5 }}
        >
          <div className="flex-shrink-0 w-7 h-7 rounded-full bg-purple-100 flex items-center justify-center text-purple-600 text-xs font-bold">
            4
          </div>
          <div className="flex-1">
            <div className="text-xs font-medium text-gray-700 mb-1">
              MLP Neural Network — Forward Pass
            </div>
            <div className="text-[10px] text-gray-500 mb-2">
              {layers.map((l: any) => l.size).join(" → ")} neurons &middot;{" "}
              {activation} activation &middot;{" "}
              {result.architecture_diagram?.total_params?.toLocaleString() ?? "?"} parameters
            </div>
            <div className="overflow-x-auto pb-1">
              <svg
                width={NET_W}
                height={NET_H}
                viewBox={`0 0 ${NET_W} ${NET_H}`}
                className="block"
              >
                {/* Connection lines between layers */}
                {layerNeurons.slice(0, -1).map((fromLayer, li) =>
                  fromLayer
                    .filter((n) => !n.ellipsis)
                    .map((from, fi) =>
                      layerNeurons[li + 1]
                        .filter((n) => !n.ellipsis)
                        .map((to, ti) => (
                          <motion.line
                            key={`c-${li}-${fi}-${ti}`}
                            x1={from.x}
                            y1={from.y}
                            x2={to.x}
                            y2={to.y}
                            stroke="#c7d2fe"
                            strokeWidth={0.6}
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 0.5 }}
                            transition={{ delay: 1.8 + li * 0.25 }}
                          />
                        ))
                    )
                )}
                {/* Animated "signal" pulses travelling through connections */}
                {layerNeurons.slice(0, -1).map((fromLayer, li) => {
                  // One pulse per layer gap, using the middle neurons
                  const from = fromLayer.filter((n) => !n.ellipsis)[0];
                  const to = layerNeurons[li + 1].filter((n) => !n.ellipsis)[0];
                  if (!from || !to) return null;
                  return (
                    <motion.circle
                      key={`pulse-${li}`}
                      r={3}
                      fill="#a855f7"
                      initial={{ cx: from.x, cy: from.y, opacity: 0 }}
                      animate={{
                        cx: [from.x, to.x],
                        cy: [from.y, to.y],
                        opacity: [0, 1, 1, 0],
                      }}
                      transition={{
                        duration: 0.8,
                        delay: 2.2 + li * 0.6,
                        ease: "easeInOut",
                      }}
                    />
                  );
                })}
                {/* Neuron circles */}
                {layerNeurons.map((neurons, li) => {
                  const layer = layers[li];
                  const fill =
                    layer?.type === "input"
                      ? "#3b82f6"
                      : layer?.type === "output"
                        ? "#22c55e"
                        : "#a855f7";
                  return neurons.map((n, ni) =>
                    n.ellipsis ? (
                      <motion.text
                        key={`n-${li}-${ni}`}
                        x={n.x}
                        y={n.y + 4}
                        textAnchor="middle"
                        fontSize={12}
                        fill="#9ca3af"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 1.7 + li * 0.25 }}
                      >
                        ⋮
                      </motion.text>
                    ) : (
                      <motion.circle
                        key={`n-${li}-${ni}`}
                        cx={n.x}
                        cy={n.y}
                        r={5}
                        fill={fill}
                        stroke="white"
                        strokeWidth={1}
                        initial={{ scale: 0, opacity: 0 }}
                        animate={{ scale: 1, opacity: 0.85 }}
                        transition={{
                          delay: 1.7 + li * 0.25 + ni * 0.02,
                          type: "spring",
                          stiffness: 300,
                        }}
                      />
                    )
                  );
                })}
                {/* Layer labels */}
                {layers.map((l: any, i: number) => (
                  <motion.text
                    key={`lbl-${i}`}
                    x={layerX[i]}
                    y={NET_H - 2}
                    textAnchor="middle"
                    fontSize={8}
                    fill="#6b7280"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 1.7 + i * 0.25 }}
                  >
                    {l.type === "input"
                      ? `Input (${l.size})`
                      : l.type === "output"
                        ? `Output (${l.size})`
                        : `Hidden (${l.size})`}
                  </motion.text>
                ))}
                {/* Activation labels between layers */}
                {layers.slice(1).map((l: any, i: number) => (
                  <motion.text
                    key={`act-${i}`}
                    x={(layerX[i] + layerX[i + 1]) / 2}
                    y={14}
                    textAnchor="middle"
                    fontSize={7}
                    fill="#a78bfa"
                    fontStyle="italic"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 0.7 }}
                    transition={{ delay: 2.0 + i * 0.25 }}
                  >
                    {l.activation || activation}
                  </motion.text>
                ))}
              </svg>
            </div>
          </div>
        </motion.div>
      )}

      {/* Step 5: Training — Loss Curve Descent */}
      {lossCurve.length > 1 && (
        <motion.div
          className="flex items-start gap-3"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 2.8 }}
        >
          <div className="flex-shrink-0 w-7 h-7 rounded-full bg-rose-100 flex items-center justify-center text-rose-600 text-xs font-bold">
            5
          </div>
          <div className="flex-1">
            <div className="text-xs font-medium text-gray-700 mb-1">
              Gradient Descent Training
            </div>
            <div className="text-[10px] text-gray-500 mb-2">
              {result.n_iterations || lossCurve.length} iterations &middot; loss:{" "}
              {lossCurve[0]?.toFixed(3)} → {lossCurve[lossCurve.length - 1]?.toFixed(3)}
              {result.training_analysis?.convergence === "converged" && (
                <span className="text-green-600 ml-1">(converged)</span>
              )}
            </div>
            <div className="overflow-x-auto">
              <svg width={320} height={90} viewBox="0 0 320 90" className="block">
                {/* Grid lines */}
                {[0, 1, 2, 3].map((i) => (
                  <line
                    key={i}
                    x1={10}
                    y1={10 + i * 22}
                    x2={310}
                    y2={10 + i * 22}
                    stroke="#f3f4f6"
                    strokeWidth={0.5}
                  />
                ))}
                {/* Loss curve */}
                <motion.path
                  d={lossPath}
                  fill="none"
                  stroke="#f43f5e"
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 2, delay: 3.0, ease: "easeOut" }}
                />
                <text x={0} y={8} fontSize={7} fill="#9ca3af">
                  loss
                </text>
                <text x={275} y={88} fontSize={7} fill="#9ca3af">
                  iterations →
                </text>
              </svg>
            </div>
          </div>
        </motion.div>
      )}

      {/* Step 6: Softmax → Classification Output */}
      <motion.div
        className="flex items-start gap-3"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 3.5 }}
      >
        <div className="flex-shrink-0 w-7 h-7 rounded-full bg-green-100 flex items-center justify-center text-green-600 text-xs font-bold">
          {lossCurve.length > 1 ? "6" : "5"}
        </div>
        <div className="flex-1">
          <div className="text-xs font-medium text-gray-700 mb-1">
            Softmax → Class Probabilities
          </div>
          <div className="text-[10px] text-gray-500 mb-2">
            Output layer produces probability for each of {classNames.length} classes
          </div>
          <div className="space-y-1.5">
            {classNames.map((name: string, i: number) => {
              const cls = perClass[name];
              const f1 = cls?.f1 ?? 0;
              const color =
                f1 >= 0.8
                  ? "bg-green-500"
                  : f1 >= 0.6
                    ? "bg-yellow-500"
                    : "bg-red-400";
              return (
                <div key={name} className="flex items-center gap-2">
                  <span className="text-[10px] text-gray-600 w-14 text-right truncate font-medium">
                    {name}
                  </span>
                  <div className="flex-1 bg-gray-100 rounded-full h-2.5 max-w-[200px]">
                    <motion.div
                      className={`h-full rounded-full ${color}`}
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.max(f1 * 100, 2)}%` }}
                      transition={{
                        duration: 0.6,
                        delay: 3.8 + i * 0.12,
                        ease: "easeOut",
                      }}
                    />
                  </div>
                  <motion.span
                    className="text-[10px] text-gray-500 w-10 font-mono"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 4.0 + i * 0.12 }}
                  >
                    {(f1 * 100).toFixed(0)}%
                  </motion.span>
                </div>
              );
            })}
          </div>
        </div>
      </motion.div>
    </div>
  );
}

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

// --- Architecture Tab ---

function ArchitectureTab({ data }: { data: any }) {
  return (
    <div className="space-y-4">
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-purple-900">
          Network Architecture
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
            const maxNeurons = Math.max(...data.layers.map((l: any) => l.size));
            const barHeight = Math.max(20, Math.min(120, (layer.size / maxNeurons) * 120));
            const bgColor =
              layer.type === "input"
                ? "bg-gradient-to-b from-blue-400 to-blue-600"
                : layer.type === "output"
                  ? "bg-gradient-to-b from-green-400 to-green-600"
                  : "bg-gradient-to-b from-purple-400 to-purple-600";

            return (
              <div key={i} className="flex items-center gap-3">
                <div className="flex flex-col items-center">
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
                  <div className="mt-2 text-center">
                    <div className="text-[10px] font-semibold text-gray-700">
                      {layer.type === "input" ? "Input" : layer.type === "output" ? "Output" : `Hidden ${i}`}
                    </div>
                    {layer.activation && (
                      <div className="text-[9px] text-gray-400">{layer.activation}</div>
                    )}
                    {layer.params && (
                      <div className="text-[9px] text-gray-400">{layer.params.toLocaleString()} params</div>
                    )}
                  </div>
                </div>
                {i < data.layers.length - 1 && <div className="text-gray-300 text-lg">→</div>}
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
              <th className="text-left px-3 py-2 text-xs text-gray-600">Layer</th>
              <th className="text-center px-3 py-2 text-xs text-gray-600">Neurons</th>
              <th className="text-center px-3 py-2 text-xs text-gray-600">Activation</th>
              <th className="text-center px-3 py-2 text-xs text-gray-600">Parameters</th>
            </tr>
          </thead>
          <tbody>
            {data.layers.map((layer: any, i: number) => (
              <tr key={i} className="border-t hover:bg-gray-50">
                <td className="px-3 py-2"><span className="font-medium">{layer.name}</span></td>
                <td className="px-3 py-2 text-center font-mono">{layer.size}</td>
                <td className="px-3 py-2 text-center text-xs">{layer.activation || "—"}</td>
                <td className="px-3 py-2 text-center font-mono text-xs">{layer.params ? layer.params.toLocaleString() : "—"}</td>
              </tr>
            ))}
          </tbody>
          <tfoot className="bg-gray-50">
            <tr className="border-t font-semibold">
              <td className="px-3 py-2" colSpan={3}>Total Parameters</td>
              <td className="px-3 py-2 text-center font-mono">{data.total_params.toLocaleString()}</td>
            </tr>
          </tfoot>
        </table>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <p className="text-xs text-blue-700">
            <strong>Parameters</strong> are the learnable weights — each connection between neurons has a weight, plus each neuron has a bias. More parameters = more capacity to learn complex patterns, but also more risk of overfitting.
          </p>
        </div>
      </div>
    </div>
  );
}

