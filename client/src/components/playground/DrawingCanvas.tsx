/**
 * DrawingCanvas — draw an image and predict using a trained model.
 * Supports mouse + touch for tablet use. Auto-resizes to model's training resolution.
 */

import { useRef, useState, useEffect, useCallback } from "react";
import { PenTool, Eraser, Loader2 } from "lucide-react";

interface Prediction {
  class_name: string;
  confidence: number;
  all_scores: { class_name: string; score: number }[];
}

interface DrawingCanvasProps {
  imageWidth: number;
  imageHeight: number;
  classNames: string[];
  onPredict: (pixels: number[]) => Promise<Prediction>;
}

const CANVAS_SIZE = 280;

export const DrawingCanvas = ({
  imageWidth,
  imageHeight,
  classNames,
  onPredict,
}: DrawingCanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [brushSize, setBrushSize] = useState(16);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [hasDrawn, setHasDrawn] = useState(false);

  // Initialize canvas with black background
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  }, []);

  const getPos = useCallback(
    (e: React.MouseEvent | React.TouchEvent): [number, number] => {
      const canvas = canvasRef.current!;
      const rect = canvas.getBoundingClientRect();
      const scaleX = CANVAS_SIZE / rect.width;
      const scaleY = CANVAS_SIZE / rect.height;
      if ("touches" in e) {
        const touch = e.touches[0];
        return [
          (touch.clientX - rect.left) * scaleX,
          (touch.clientY - rect.top) * scaleY,
        ];
      }
      return [
        (e.clientX - rect.left) * scaleX,
        (e.clientY - rect.top) * scaleY,
      ];
    },
    [],
  );

  const startDraw = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      e.preventDefault();
      setIsDrawing(true);
      setHasDrawn(true);
      const canvas = canvasRef.current!;
      const ctx = canvas.getContext("2d")!;
      const [x, y] = getPos(e);
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.lineWidth = brushSize;
      ctx.strokeStyle = "#fff";
      // Draw a dot for single click
      ctx.lineTo(x + 0.1, y + 0.1);
      ctx.stroke();
    },
    [getPos, brushSize],
  );

  const draw = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      e.preventDefault();
      if (!isDrawing) return;
      const canvas = canvasRef.current!;
      const ctx = canvas.getContext("2d")!;
      const [x, y] = getPos(e);
      ctx.lineWidth = brushSize;
      ctx.strokeStyle = "#fff";
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.lineTo(x, y);
      ctx.stroke();
    },
    [isDrawing, getPos, brushSize],
  );

  const endDraw = useCallback(() => {
    setIsDrawing(false);
  }, []);

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    setPrediction(null);
    setHasDrawn(false);
  }, []);

  // Downsample drawn image to model resolution → grayscale pixel array
  const getPixels = useCallback((): number[] => {
    const canvas = canvasRef.current!;
    const resized = document.createElement("canvas");
    resized.width = imageWidth;
    resized.height = imageHeight;
    const ctx = resized.getContext("2d")!;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(canvas, 0, 0, imageWidth, imageHeight);
    const data = ctx.getImageData(0, 0, imageWidth, imageHeight).data;
    const pixels: number[] = [];
    for (let i = 0; i < data.length; i += 4) {
      pixels.push(
        Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]),
      );
    }
    return pixels;
  }, [imageWidth, imageHeight]);

  const handlePredict = useCallback(async () => {
    if (!hasDrawn) return;
    setPredicting(true);
    try {
      const pixels = getPixels();
      const result = await onPredict(pixels);
      setPrediction(result);
    } catch (err) {
      console.error("Prediction failed:", err);
    } finally {
      setPredicting(false);
    }
  }, [hasDrawn, getPixels, onPredict]);

  const topScore = prediction
    ? Math.max(...prediction.all_scores.map((s) => s.score))
    : 0;

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row gap-4">
        {/* Drawing area */}
        <div className="flex flex-col items-center gap-3">
          <div className="rounded-xl border-2 border-slate-300 bg-black overflow-hidden shadow-lg">
            <canvas
              ref={canvasRef}
              width={CANVAS_SIZE}
              height={CANVAS_SIZE}
              style={{ width: 280, height: 280, touchAction: "none", cursor: "crosshair" }}
              onMouseDown={startDraw}
              onMouseMove={draw}
              onMouseUp={endDraw}
              onMouseLeave={endDraw}
              onTouchStart={startDraw}
              onTouchMove={draw}
              onTouchEnd={endDraw}
            />
          </div>

          {/* Controls */}
          <div className="flex items-center gap-3 w-full max-w-[280px]">
            <PenTool className="w-4 h-4 text-slate-400 shrink-0" />
            <input
              type="range"
              min={4}
              max={36}
              value={brushSize}
              onChange={(e) => setBrushSize(Number(e.target.value))}
              className="flex-1 accent-violet-600"
            />
            <span className="text-xs text-slate-500 w-8 text-right">{brushSize}px</span>
          </div>

          <div className="flex gap-2 w-full max-w-[280px]">
            <button
              onClick={clearCanvas}
              className="flex-1 flex items-center justify-center gap-2 py-2 border border-slate-300 rounded-lg text-sm text-slate-600 hover:bg-slate-50 transition-colors"
            >
              <Eraser className="w-4 h-4" />
              Clear
            </button>
            <button
              onClick={handlePredict}
              disabled={!hasDrawn || predicting}
              className="flex-1 flex items-center justify-center gap-2 py-2 bg-violet-600 hover:bg-violet-700 disabled:bg-slate-200 disabled:text-slate-400 text-white rounded-lg text-sm font-semibold transition-colors"
            >
              {predicting ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <PenTool className="w-4 h-4" />
              )}
              {predicting ? "Predicting…" : "Predict"}
            </button>
          </div>

          <p className="text-[10px] text-slate-400 text-center">
            Draw with white on black. Model input: {imageWidth}×{imageHeight}px grayscale
          </p>
        </div>

        {/* Prediction result */}
        <div className="flex-1 min-w-[200px]">
          {prediction ? (
            <div className="space-y-3">
              {/* Main prediction */}
              <div className="rounded-xl border border-violet-200 bg-violet-50/60 p-4 text-center">
                <p className="text-xs font-medium text-violet-500 uppercase tracking-wider mb-1">
                  Prediction
                </p>
                <p className="text-4xl font-black text-violet-700">
                  {prediction.class_name}
                </p>
                <p className="text-sm text-violet-600 mt-1">
                  {(prediction.confidence * 100).toFixed(1)}% confidence
                </p>
              </div>

              {/* All class scores */}
              <div className="space-y-1.5">
                <p className="text-xs font-semibold text-slate-600">All Class Scores</p>
                {prediction.all_scores
                  .sort((a, b) => b.score - a.score)
                  .map((s) => (
                    <div key={s.class_name} className="flex items-center gap-2">
                      <span className="text-xs text-slate-600 w-16 truncate text-right font-medium">
                        {s.class_name}
                      </span>
                      <div className="flex-1 h-4 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all ${
                            s.class_name === prediction.class_name
                              ? "bg-violet-500"
                              : "bg-slate-300"
                          }`}
                          style={{
                            width: `${topScore > 0 ? (s.score / topScore) * 100 : 0}%`,
                          }}
                        />
                      </div>
                      <span className="text-[10px] text-slate-500 w-10 text-right">
                        {(s.score * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          ) : (
            <div className="rounded-xl border-2 border-dashed border-slate-200 bg-slate-50/50 p-8 text-center">
              <PenTool className="w-8 h-8 text-slate-300 mx-auto mb-2" />
              <p className="text-sm font-medium text-slate-500">Draw something!</p>
              <p className="text-xs text-slate-400 mt-1">
                Draw a {classNames.length > 0 ? classNames.join(", ") : "character"} on the
                canvas, then click Predict
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
