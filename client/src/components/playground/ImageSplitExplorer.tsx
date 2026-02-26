/**
 * ImageSplitExplorer - Interactive learning for Image Split node.
 * Tabs: Split Preview | Class Balance | Split Ratios | Quiz
 */

import { useState } from "react";
import {
  SplitSquareVertical,
  BarChart3,
  PieChart,
  HelpCircle,
  Info,
  CheckCircle,
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
        <span className="text-[10px] text-gray-600 truncate max-w-[80px]">
          {label}
        </span>
      )}
    </div>
  );
}

// --- Main Explorer ---

interface ImageSplitExplorerProps {
  result: any;
}

type Tab = "split_viz" | "balance" | "ratios" | "quiz";

export const ImageSplitExplorer = ({
  result,
}: ImageSplitExplorerProps) => {
  const [activeTab, setActiveTab] = useState<Tab>("split_viz");

  const tabs: { id: Tab; label: string; icon: any; available: boolean }[] = [
    {
      id: "split_viz",
      label: "Split Preview",
      icon: SplitSquareVertical,
      available: !!result.split_visualization,
    },
    {
      id: "balance",
      label: "Class Balance",
      icon: BarChart3,
      available: !!result.class_balance,
    },
    {
      id: "ratios",
      label: "Split Ratios",
      icon: PieChart,
      available: !!result.split_ratios,
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

      {activeTab === "split_viz" && result.split_visualization && (
        <SplitVizTab data={result.split_visualization} />
      )}
      {activeTab === "balance" && result.class_balance && (
        <ClassBalanceTab
          data={result.class_balance}
          stratified={result.split_ratios?.stratified}
        />
      )}
      {activeTab === "ratios" && result.split_ratios && (
        <SplitRatiosTab data={result.split_ratios} />
      )}
      {activeTab === "quiz" && result.quiz_questions && (
        <QuizTab questions={result.quiz_questions} />
      )}
    </div>
  );
};

// --- Split Visualization Tab (per-class train vs test) ---

function SplitVizTab({ data }: { data: any }) {
  const [selectedClass, setSelectedClass] = useState<number>(0);
  const classes = data.classes || [];
  const cls = classes[selectedClass];

  const totalTrain = classes.reduce(
    (sum: number, c: any) => sum + (c.train_count || 0),
    0
  );
  const totalTest = classes.reduce(
    (sum: number, c: any) => sum + (c.test_count || 0),
    0
  );

  return (
    <div className="space-y-4">
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-purple-900">
          Split Preview
        </h3>
        <p className="text-xs text-purple-700 mt-1">
          {totalTrain} training + {totalTest} test images across{" "}
          {classes.length} classes. No image appears in both sets!
        </p>
      </div>

      {/* Class selector pills */}
      {classes.length > 1 && (
        <div className="flex flex-wrap gap-2">
          {classes.map((c: any, i: number) => (
            <button
              key={i}
              onClick={() => setSelectedClass(i)}
              className={`px-3 py-1.5 text-xs rounded-lg border font-medium transition-all ${
                selectedClass === i
                  ? "bg-blue-600 text-white border-blue-600 shadow-md"
                  : "bg-white text-gray-700 border-gray-300 hover:border-blue-400"
              }`}
            >
              {c.class_name}
              <span className="ml-1 opacity-75">
                ({c.train_count}+{c.test_count})
              </span>
            </button>
          ))}
        </div>
      )}

      {cls && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Train samples */}
          <div className="bg-white border border-green-200 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-green-700 mb-3 flex items-center justify-between">
              <span>Training Set</span>
              <span className="text-xs font-normal bg-green-100 text-green-600 px-2 py-0.5 rounded-full">
                {cls.train_count} images
              </span>
            </h4>
            <div className="flex flex-wrap gap-2">
              {cls.train_samples?.length > 0 ? (
                cls.train_samples.map((pixels: number[], i: number) => (
                  <PixelImage
                    key={i}
                    pixels={pixels}
                    width={data.width}
                    height={data.height}
                    size={64}
                    label={cls.class_name}
                  />
                ))
              ) : (
                <p className="text-xs text-gray-400">No samples available</p>
              )}
            </div>
          </div>

          {/* Test samples */}
          <div className="bg-white border border-blue-200 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-blue-700 mb-3 flex items-center justify-between">
              <span>Test Set</span>
              <span className="text-xs font-normal bg-blue-100 text-blue-600 px-2 py-0.5 rounded-full">
                {cls.test_count} images
              </span>
            </h4>
            <div className="flex flex-wrap gap-2">
              {cls.test_samples?.length > 0 ? (
                cls.test_samples.map((pixels: number[], i: number) => (
                  <PixelImage
                    key={i}
                    pixels={pixels}
                    width={data.width}
                    height={data.height}
                    size={64}
                    label={cls.class_name}
                  />
                ))
              ) : (
                <p className="text-xs text-gray-400">No samples available</p>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <p className="text-xs text-blue-700">
            <strong>Why split?</strong> The model learns from training data and
            is evaluated on test data it has never seen. This measures how well
            the model <em>generalizes</em> to new images, rather than just
            memorizing the training set.
          </p>
        </div>
      </div>
    </div>
  );
}

// --- Class Balance Tab ---

function ClassBalanceTab({
  data,
  stratified,
}: {
  data: any;
  stratified?: boolean;
}) {
  return (
    <div className="space-y-4">
      <div className="bg-green-50 border border-green-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-green-900">
          Class Balance: Train vs Test
        </h3>
        <p className="text-xs text-green-700 mt-1">
          {stratified
            ? "Stratified split — class proportions are preserved in both sets."
            : "Random split — class proportions may differ between sets."}
        </p>
      </div>

      {/* Side-by-side comparison table */}
      <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="text-left px-3 py-2 text-xs text-gray-600">
                Class
              </th>
              <th className="text-center px-3 py-2 text-xs text-green-600">
                Train
              </th>
              <th className="text-center px-3 py-2 text-xs text-blue-600">
                Test
              </th>
              <th className="text-center px-3 py-2 text-xs text-gray-600">
                Diff
              </th>
            </tr>
          </thead>
          <tbody>
            {data.classes?.map((cls: any, i: number) => {
              const diff = Math.abs(
                cls.train_pct - cls.test_pct
              ).toFixed(1);
              return (
                <tr key={i} className="border-t hover:bg-gray-50">
                  <td className="px-3 py-2 font-medium text-gray-900">
                    {cls.class_name}
                  </td>
                  <td className="px-3 py-2 text-center">
                    <div className="flex items-center gap-2 justify-center">
                      <div className="w-16 bg-gray-100 rounded-full h-2">
                        <div
                          className="bg-green-500 h-full rounded-full"
                          style={{ width: `${cls.train_pct}%` }}
                        />
                      </div>
                      <span className="text-xs">
                        {cls.train_count} ({cls.train_pct}%)
                      </span>
                    </div>
                  </td>
                  <td className="px-3 py-2 text-center">
                    <div className="flex items-center gap-2 justify-center">
                      <div className="w-16 bg-gray-100 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-full rounded-full"
                          style={{ width: `${cls.test_pct}%` }}
                        />
                      </div>
                      <span className="text-xs">
                        {cls.test_count} ({cls.test_pct}%)
                      </span>
                    </div>
                  </td>
                  <td
                    className={`px-3 py-2 text-center text-xs font-mono ${
                      Number(diff) < 1
                        ? "text-green-600"
                        : "text-orange-600"
                    }`}
                  >
                    ±{diff}%
                  </td>
                </tr>
              );
            })}
          </tbody>
          <tfoot className="bg-gray-50 border-t">
            <tr className="font-medium text-xs text-gray-600">
              <td className="px-3 py-2">Total</td>
              <td className="px-3 py-2 text-center">
                {data.total_train} images
              </td>
              <td className="px-3 py-2 text-center">
                {data.total_test} images
              </td>
              <td className="px-3 py-2 text-center">—</td>
            </tr>
          </tfoot>
        </table>
      </div>

      {stratified && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-3 flex items-center gap-2">
          <CheckCircle className="w-5 h-5 text-green-600" />
          <p className="text-xs text-green-700">
            <strong>Stratified split</strong> ensures that the ratio of each
            class in the training set matches the test set. This prevents
            situations where one class is overrepresented in training and
            underrepresented in testing.
          </p>
        </div>
      )}
    </div>
  );
}

// --- Split Ratios Tab ---

function SplitRatiosTab({ data }: { data: any }) {
  const trainPct = Math.round(data.train_pct);
  const testPct = Math.round(data.test_pct);

  return (
    <div className="space-y-4">
      <div className="bg-orange-50 border border-orange-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-orange-900">
          Split Ratios
        </h3>
        <p className="text-xs text-orange-700 mt-1">
          Your data was split into {trainPct}% training and {testPct}% testing.
        </p>
      </div>

      {/* Visual ratio bar */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex h-8 rounded-full overflow-hidden mb-4">
          <div
            className="bg-green-500 flex items-center justify-center text-white text-xs font-bold"
            style={{ width: `${trainPct}%` }}
          >
            Train {trainPct}%
          </div>
          <div
            className="bg-blue-500 flex items-center justify-center text-white text-xs font-bold"
            style={{ width: `${testPct}%` }}
          >
            Test {testPct}%
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="bg-green-50 border border-green-200 rounded-lg p-3 text-center">
            <div className="text-xs text-green-500">Training Images</div>
            <div className="text-2xl font-bold text-green-700">
              {(data.train_count ?? 0).toLocaleString()}
            </div>
          </div>
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-center">
            <div className="text-xs text-blue-500">Test Images</div>
            <div className="text-2xl font-bold text-blue-700">
              {(data.test_count ?? 0).toLocaleString()}
            </div>
          </div>
        </div>
      </div>

      {/* Common ratios comparison */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-800 mb-3">
          Common Split Ratios
        </h4>
        <div className="space-y-2">
          {[
            { label: "80/20 (Standard)", train: 80, test: 20 },
            { label: "70/30 (Conservative)", train: 70, test: 30 },
            { label: "90/10 (Data-rich)", train: 90, test: 10 },
          ].map((ratio, i) => {
            const isCurrent = Math.abs(ratio.train - trainPct) < 3;
            return (
              <div
                key={i}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg ${
                  isCurrent
                    ? "bg-blue-50 border border-blue-300"
                    : "bg-gray-50"
                }`}
              >
                <div className="w-28 text-xs font-medium text-gray-700">
                  {ratio.label}
                </div>
                <div className="flex-1 flex h-3 rounded-full overflow-hidden">
                  <div
                    className="bg-green-400 h-full"
                    style={{ width: `${ratio.train}%` }}
                  />
                  <div
                    className="bg-blue-400 h-full"
                    style={{ width: `${ratio.test}%` }}
                  />
                </div>
                {isCurrent && (
                  <span className="text-[10px] text-blue-600 font-semibold">
                    Your split
                  </span>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
