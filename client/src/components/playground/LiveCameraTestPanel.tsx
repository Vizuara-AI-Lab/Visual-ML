/**
 * LiveCameraTestPanel — run a trained image model on live camera frames.
 * Captures a frame, sends pixels to the backend predict endpoint, and shows result.
 */

import { useRef, useState, useEffect, useCallback } from "react";
import { Video, VideoOff, Camera, Zap, AlertCircle } from "lucide-react";

interface Prediction {
  class_name: string;
  confidence: number;
  all_scores: { class_name: string; score: number }[];
}

interface LiveCameraTestPanelProps {
  modelId: string;
  modelPath: string;
  classNames: string[];
  imageWidth: number;
  imageHeight: number;
  onPredict: (pixels: number[]) => Promise<Prediction>;
}

function parseSize(w: number, h: number) {
  return [w || 28, h || 28];
}

function capturePixels(
  video: HTMLVideoElement,
  width: number,
  height: number,
): { pixels: number[]; dataUrl: string } {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, width, height);

  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const side = Math.min(vw, vh);
  const sx = (vw - side) / 2;
  const sy = (vh - side) / 2;
  ctx.drawImage(video, sx, sy, side, side, 0, 0, width, height);

  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  const pixels: number[] = [];
  for (let i = 0; i < data.length; i += 4) {
    const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    pixels.push(gray);
    data[i] = gray;
    data[i + 1] = gray;
    data[i + 2] = gray;
  }
  ctx.putImageData(imageData, 0, 0);

  // Preview at larger size
  const preview = document.createElement("canvas");
  preview.width = 112;
  preview.height = 112;
  const pctx = preview.getContext("2d")!;
  pctx.imageSmoothingEnabled = false;
  pctx.drawImage(canvas, 0, 0, 112, 112);

  return { pixels, dataUrl: preview.toDataURL("image/png") };
}

export const LiveCameraTestPanel = ({
  modelId,
  classNames,
  imageWidth,
  imageHeight,
  onPredict,
}: LiveCameraTestPanelProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const autoRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [lastCapture, setLastCapture] = useState<string | null>(null);
  const [autoMode, setAutoMode] = useState(false);

  const [tw, th] = parseSize(imageWidth, imageHeight);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setCameraError(`Camera access denied: ${msg}`);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (autoRef.current) clearInterval(autoRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
    setAutoMode(false);
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const runPrediction = useCallback(async () => {
    if (!videoRef.current || !cameraActive || predicting) return;
    setPredicting(true);
    const { pixels, dataUrl } = capturePixels(videoRef.current, tw, th);
    setLastCapture(dataUrl);
    try {
      const result = await onPredict(pixels);
      setPrediction(result);
    } catch {
      // silently ignore mid-stream errors
    } finally {
      setPredicting(false);
    }
  }, [cameraActive, predicting, tw, th, onPredict]);

  // Auto mode: predict every 1.5 s
  useEffect(() => {
    if (autoMode && cameraActive) {
      autoRef.current = setInterval(() => {
        runPrediction();
      }, 1500);
    } else if (autoRef.current) {
      clearInterval(autoRef.current);
      autoRef.current = null;
    }
    return () => {
      if (autoRef.current) clearInterval(autoRef.current);
    };
  }, [autoMode, cameraActive, runPrediction]);

  if (!modelId) {
    return (
      <div className="rounded-xl border border-amber-200 bg-amber-50/60 p-4 flex items-start gap-3">
        <AlertCircle className="w-4 h-4 text-amber-500 mt-0.5 shrink-0" />
        <div>
          <p className="text-sm font-semibold text-amber-800">No Trained Model</p>
          <p className="text-xs text-amber-700 mt-0.5">
            Run the full pipeline (including CNN Classifier) first, then open this panel again.
          </p>
        </div>
      </div>
    );
  }

  const topScore = prediction?.all_scores?.[0];

  return (
    <div className="space-y-4">
      {/* Camera preview */}
      <div className="rounded-xl border border-slate-200 bg-slate-900 overflow-hidden">
        <div className="relative aspect-video flex items-center justify-center">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className={`w-full h-full object-cover ${cameraActive ? "opacity-100" : "opacity-0"}`}
          />
          {!cameraActive && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
              <VideoOff className="w-10 h-10 text-slate-500" />
              <p className="text-sm text-slate-400">Camera not started</p>
            </div>
          )}

          {/* Live prediction overlay */}
          {cameraActive && prediction && (
            <div className="absolute bottom-3 left-3 right-3 bg-black/70 backdrop-blur-sm rounded-xl p-3">
              <div className="flex items-center gap-3">
                {lastCapture && (
                  <img
                    src={lastCapture}
                    alt="captured"
                    className="w-10 h-10 rounded-md border border-white/20"
                    style={{ imageRendering: "pixelated" }}
                  />
                )}
                <div className="flex-1 min-w-0">
                  <p className="text-white font-bold text-base truncate">
                    {prediction.class_name}
                  </p>
                  <p className="text-slate-300 text-xs">
                    {Math.round(prediction.confidence * 100)}% confidence
                  </p>
                </div>
                {predicting && (
                  <div className="w-4 h-4 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
                )}
              </div>
            </div>
          )}

          {/* Square guide */}
          {cameraActive && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="border-2 border-white/30 rounded-lg"
                style={{ width: "60%", aspectRatio: "1/1" }} />
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2 p-3 bg-slate-800">
          {!cameraActive ? (
            <button
              onClick={startCamera}
              className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-700 text-white rounded-lg text-sm font-medium transition-colors"
            >
              <Video className="w-4 h-4" />
              Start Camera
            </button>
          ) : (
            <>
              <button
                onClick={runPrediction}
                disabled={predicting || autoMode}
                className="flex items-center gap-2 px-4 py-2 bg-white hover:bg-slate-100 text-slate-900 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
              >
                <Camera className="w-4 h-4" />
                Predict
              </button>
              <button
                onClick={() => setAutoMode((v) => !v)}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  autoMode
                    ? "bg-violet-600 text-white"
                    : "bg-slate-700 hover:bg-slate-600 text-slate-200"
                }`}
              >
                <Zap className="w-4 h-4" />
                {autoMode ? "Live ON" : "Live"}
              </button>
              <button
                onClick={stopCamera}
                className="ml-auto p-2 bg-red-900/60 hover:bg-red-900 text-red-300 rounded-lg transition-colors"
              >
                <VideoOff className="w-4 h-4" />
              </button>
            </>
          )}
          {cameraError && (
            <p className="text-xs text-red-400 ml-2">{cameraError}</p>
          )}
        </div>
      </div>

      {/* Scores breakdown */}
      {prediction && (
        <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-3">
          <p className="text-xs font-bold text-slate-500 uppercase tracking-wider">
            All Class Scores
          </p>
          <div className="space-y-2">
            {prediction.all_scores.map((s) => {
              const pct = Math.round(s.score * 100);
              const isTop = s.class_name === prediction.class_name;
              return (
                <div key={s.class_name}>
                  <div className="flex justify-between text-xs mb-0.5">
                    <span className={`font-semibold ${isTop ? "text-violet-700" : "text-slate-600"}`}>
                      {s.class_name}
                    </span>
                    <span className={isTop ? "text-violet-600 font-bold" : "text-slate-400"}>
                      {pct}%
                    </span>
                  </div>
                  <div className="h-1.5 rounded-full bg-slate-100 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${isTop ? "bg-violet-500" : "bg-slate-300"}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          {topScore && topScore.score < 0.5 && (
            <p className="text-xs text-amber-600 flex items-center gap-1.5">
              <AlertCircle className="w-3.5 h-3.5 shrink-0" />
              Low confidence — try better lighting or a different angle
            </p>
          )}
        </div>
      )}

      {/* Classes info */}
      <div className="rounded-lg bg-slate-50 border border-slate-100 p-3">
        <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2">
          Model recognizes {classNames.length} classes
        </p>
        <div className="flex flex-wrap gap-1">
          {classNames.map((cls) => (
            <span
              key={cls}
              className="px-2 py-0.5 bg-white rounded-md text-[11px] text-slate-600 border border-slate-200 font-medium"
            >
              {cls}
            </span>
          ))}
        </div>
        <p className="text-[10px] text-slate-400 mt-2 font-mono">
          Input size: {tw}×{th}px grayscale
        </p>
      </div>
    </div>
  );
};
