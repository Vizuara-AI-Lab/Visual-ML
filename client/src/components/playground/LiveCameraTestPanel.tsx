/**
 * LiveCameraTestPanel — Premium detection HUD with real-time inference.
 *
 * Uses a continuous prediction loop: as soon as one prediction finishes,
 * the next frame is captured and sent immediately. This gives the fastest
 * possible frame rate (limited only by network + inference latency).
 *
 * UI features: scanning corners, sweep line, confidence ring, prediction
 * history timeline, color-coded FPS, session stats.
 */

import { useRef, useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import {
  Video,
  VideoOff,
  Camera,
  Zap,
  ZapOff,
  AlertCircle,
  Activity,
  Clock,
  Hash,
} from "lucide-react";

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

/** Capture a single frame, convert to grayscale pixels + preview data URL. */
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
    const gray = Math.round(
      0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2],
    );
    pixels.push(gray);
    data[i] = gray;
    data[i + 1] = gray;
    data[i + 2] = gray;
  }
  ctx.putImageData(imageData, 0, 0);

  const preview = document.createElement("canvas");
  preview.width = 112;
  preview.height = 112;
  const pctx = preview.getContext("2d")!;
  pctx.imageSmoothingEnabled = false;
  pctx.drawImage(canvas, 0, 0, 112, 112);

  return { pixels, dataUrl: preview.toDataURL("image/png") };
}

/** Animated SVG confidence ring gauge. */
function ConfidenceRing({
  value,
  size = 110,
  label,
}: {
  value: number;
  size?: number;
  label?: string;
}) {
  const r = (size - 14) / 2;
  const circ = 2 * Math.PI * r;
  const color =
    value >= 0.8 ? "#22c55e" : value >= 0.5 ? "#eab308" : "#ef4444";
  const bgColor =
    value >= 0.8
      ? "rgba(34,197,94,0.1)"
      : value >= 0.5
        ? "rgba(234,179,8,0.1)"
        : "rgba(239,68,68,0.1)";

  return (
    <div className="flex flex-col items-center gap-1.5">
      <div className="relative" style={{ width: size, height: size }}>
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          className="block"
        >
          {/* Background glow */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r + 2}
            fill={bgColor}
          />
          {/* Track */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke="rgba(255,255,255,0.08)"
            strokeWidth="7"
          />
          {/* Progress arc */}
          <motion.circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke={color}
            strokeWidth="7"
            strokeLinecap="round"
            strokeDasharray={circ}
            animate={{ strokeDashoffset: circ * (1 - value) }}
            transition={{ duration: 0.4, ease: "easeOut" }}
            transform={`rotate(-90 ${size / 2} ${size / 2})`}
            style={{ filter: `drop-shadow(0 0 6px ${color}60)` }}
          />
        </svg>
        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span
            className="text-2xl font-black text-white tabular-nums"
            key={Math.round(value * 100)}
            initial={{ scale: 1.15, opacity: 0.7 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.2 }}
          >
            {Math.round(value * 100)}%
          </motion.span>
        </div>
      </div>
      {label && (
        <motion.span
          className="text-sm font-bold text-white truncate max-w-[120px] text-center"
          key={label}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
        >
          {label}
        </motion.span>
      )}
    </div>
  );
}

/** Format seconds into M:SS */
function formatDuration(ms: number): string {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${sec.toString().padStart(2, "0")}`;
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
  const liveLoopRef = useRef(false);
  const predictingRef = useRef(false);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [lastCapture, setLastCapture] = useState<string | null>(null);
  const [liveMode, setLiveMode] = useState(false);
  const [fps, setFps] = useState(0);
  const [totalPredictions, setTotalPredictions] = useState(0);
  const [sessionStart] = useState(Date.now());
  const [elapsed, setElapsed] = useState(0);

  // FPS tracking
  const fpsTimestamps = useRef<number[]>([]);

  const [tw, th] = parseSize(imageWidth, imageHeight);

  // Session timer
  useEffect(() => {
    if (!cameraActive) return;
    const timer = setInterval(() => setElapsed(Date.now() - sessionStart), 1000);
    return () => clearInterval(timer);
  }, [cameraActive, sessionStart]);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
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
    liveLoopRef.current = false;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
    setLiveMode(false);
    setFps(0);
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const updateFps = useCallback(() => {
    const now = performance.now();
    fpsTimestamps.current.push(now);
    const cutoff = now - 2000;
    fpsTimestamps.current = fpsTimestamps.current.filter((t) => t > cutoff);
    const currentFps =
      fpsTimestamps.current.length > 1
        ? Math.round(
            ((fpsTimestamps.current.length - 1) /
              (now - fpsTimestamps.current[0])) *
              1000,
          )
        : 0;
    setFps(currentFps);
  }, []);

  /** Run a single prediction (manual or called from live loop). */
  const runOnce = useCallback(async () => {
    if (!videoRef.current || !cameraActive || predictingRef.current) return;
    predictingRef.current = true;
    setPredicting(true);

    const { pixels, dataUrl } = capturePixels(videoRef.current, tw, th);
    setLastCapture(dataUrl);

    try {
      const result = await onPredict(pixels);
      setPrediction(result);
      setTotalPredictions((n) => n + 1);

      updateFps();
    } catch {
      // silently ignore mid-stream errors
    } finally {
      predictingRef.current = false;
      setPredicting(false);
    }
  }, [cameraActive, tw, th, onPredict, updateFps]);

  /**
   * Continuous live loop — fires next prediction immediately after
   * the previous one completes. No fixed interval, no wasted time.
   */
  useEffect(() => {
    if (!liveMode || !cameraActive) {
      liveLoopRef.current = false;
      return;
    }

    liveLoopRef.current = true;
    let cancelled = false;

    const loop = async () => {
      while (liveLoopRef.current && !cancelled) {
        if (!videoRef.current || !cameraActive) break;

        predictingRef.current = true;
        setPredicting(true);

        const { pixels, dataUrl } = capturePixels(videoRef.current, tw, th);
        setLastCapture(dataUrl);

        try {
          const result = await onPredict(pixels);
          if (cancelled) break;
          setPrediction(result);
          setTotalPredictions((n) => n + 1);
    
          updateFps();
        } catch {
          // ignore errors, keep looping
        } finally {
          predictingRef.current = false;
          setPredicting(false);
        }

        // Tiny yield to let React re-render
        await new Promise((r) => setTimeout(r, 30));
      }
    };

    loop();

    return () => {
      cancelled = true;
      liveLoopRef.current = false;
    };
  }, [liveMode, cameraActive, tw, th, onPredict, updateFps]);

  if (!modelId) {
    return (
      <div className="rounded-xl border border-amber-200 bg-amber-50/60 p-4 flex items-start gap-3">
        <AlertCircle className="w-4 h-4 text-amber-500 mt-0.5 shrink-0" />
        <div>
          <p className="text-sm font-semibold text-amber-800">
            No Trained Model
          </p>
          <p className="text-xs text-amber-700 mt-0.5">
            Run the full pipeline first, then open this panel to test with your
            camera.
          </p>
        </div>
      </div>
    );
  }

  const confidencePct = prediction
    ? Math.round(prediction.confidence * 100)
    : 0;
  const fpsColor =
    fps >= 3
      ? "text-green-400"
      : fps >= 1
        ? "text-yellow-400"
        : "text-red-400";

  return (
    <div className="space-y-3">
      {/* ── Two-Column Layout: Camera (left) + Status (right) ── */}
      <div className="flex gap-3">
        {/* ── Left: Camera Feed ── */}
        <div className="w-[400px] flex-shrink-0 rounded-xl border border-slate-700/50 bg-slate-950 overflow-hidden shadow-2xl shadow-black/40">
          <div
            className="relative flex items-center justify-center"
            style={{ height: 300 }}
          >
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`h-full object-cover transition-opacity duration-500 ${cameraActive ? "opacity-100" : "opacity-0"}`}
            />

            {/* Idle state */}
            {!cameraActive && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-gradient-to-b from-slate-900 to-slate-950">
                <div className="relative">
                  <VideoOff className="w-8 h-8 text-slate-600" />
                  <div className="absolute -inset-3 rounded-full border border-slate-800 animate-pulse" />
                </div>
                <p className="text-xs text-slate-500 font-medium">
                  Camera offline
                </p>
              </div>
            )}

            {/* ── HUD Scanning Corners ── */}
            {cameraActive && (
              <>
                <div
                  className={`absolute top-3 left-3 w-7 h-7 border-t-2 border-l-2 border-cyan-400/50 rounded-tl transition-opacity ${liveMode ? "opacity-100" : "opacity-40"}`}
                  style={
                    liveMode
                      ? {
                          animation: "pulse 2s ease-in-out infinite",
                          filter: "drop-shadow(0 0 4px rgba(34,211,238,0.3))",
                        }
                      : undefined
                  }
                />
                <div
                  className={`absolute top-3 right-3 w-7 h-7 border-t-2 border-r-2 border-cyan-400/50 rounded-tr transition-opacity ${liveMode ? "opacity-100" : "opacity-40"}`}
                  style={
                    liveMode
                      ? {
                          animation: "pulse 2s ease-in-out infinite 0.5s",
                          filter: "drop-shadow(0 0 4px rgba(34,211,238,0.3))",
                        }
                      : undefined
                  }
                />
                <div
                  className={`absolute bottom-3 left-3 w-7 h-7 border-b-2 border-l-2 border-cyan-400/50 rounded-bl transition-opacity ${liveMode ? "opacity-100" : "opacity-40"}`}
                  style={
                    liveMode
                      ? {
                          animation: "pulse 2s ease-in-out infinite 1s",
                          filter: "drop-shadow(0 0 4px rgba(34,211,238,0.3))",
                        }
                      : undefined
                  }
                />
                <div
                  className={`absolute bottom-3 right-3 w-7 h-7 border-b-2 border-r-2 border-cyan-400/50 rounded-br transition-opacity ${liveMode ? "opacity-100" : "opacity-40"}`}
                  style={
                    liveMode
                      ? {
                          animation: "pulse 2s ease-in-out infinite 1.5s",
                          filter: "drop-shadow(0 0 4px rgba(34,211,238,0.3))",
                        }
                      : undefined
                  }
                />
              </>
            )}

            {/* ── Scanning Line (live mode only) ── */}
            {cameraActive && liveMode && (
              <div className="absolute inset-0 pointer-events-none overflow-hidden">
                <div
                  className="absolute left-0 right-0 h-[2px]"
                  style={{
                    background:
                      "linear-gradient(90deg, transparent 0%, rgba(34,211,238,0.4) 20%, rgba(34,211,238,0.6) 50%, rgba(34,211,238,0.4) 80%, transparent 100%)",
                    boxShadow: "0 0 12px 2px rgba(34,211,238,0.15)",
                    animation: "scanline 2.5s linear infinite",
                  }}
                />
              </div>
            )}

            {/* ── LIVE badge ── */}
            {cameraActive && liveMode && (
              <div className="absolute top-2 left-2 flex items-center gap-1 bg-red-600/90 text-white px-2 py-0.5 rounded text-[9px] font-bold uppercase tracking-widest shadow-lg shadow-red-600/20">
                <span className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
                Live
              </div>
            )}

            {/* ── Compact prediction overlay on video ── */}
            {cameraActive && prediction && (
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent pt-6 pb-2 px-3">
                <div className="flex items-center gap-2">
                  <motion.p
                    className="text-white font-bold text-sm truncate"
                    key={prediction.class_name}
                    initial={{ opacity: 0, x: -4 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.15 }}
                  >
                    {prediction.class_name}
                  </motion.p>
                  <span
                    className={`px-1.5 py-0.5 rounded-full text-[10px] font-bold tabular-nums ${
                      confidencePct >= 80
                        ? "bg-green-500/20 text-green-300"
                        : confidencePct >= 50
                          ? "bg-yellow-500/20 text-yellow-300"
                          : "bg-red-500/20 text-red-300"
                    }`}
                  >
                    {confidencePct}%
                  </span>
                  {predicting && (
                    <div className="w-2.5 h-2.5 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin shrink-0" />
                  )}
                </div>
              </div>
            )}

            {/* Square guide (no prediction yet) */}
            {cameraActive && !prediction && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div
                  className="border-2 border-dashed border-white/20 rounded-lg"
                  style={{ width: "60%", aspectRatio: "1/1" }}
                />
              </div>
            )}
          </div>

          {/* ── Controls Bar ── */}
          <div className="flex items-center gap-1.5 px-2 py-2 bg-slate-900 border-t border-slate-800">
            {!cameraActive ? (
              <button
                onClick={startCamera}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-violet-600 hover:bg-violet-500 text-white rounded-lg text-xs font-semibold transition-all shadow-lg shadow-violet-600/20"
              >
                <Video className="w-3.5 h-3.5" />
                Start
              </button>
            ) : (
              <>
                <button
                  onClick={runOnce}
                  disabled={predicting || liveMode}
                  className="flex items-center gap-1 px-2.5 py-1.5 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-xs font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <Camera className="w-3.5 h-3.5" />
                  Snap
                </button>
                <button
                  onClick={() => {
                    setLiveMode((v) => !v);
                    fpsTimestamps.current = [];
                    setFps(0);
                  }}
                  className={`flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    liveMode
                      ? "bg-red-600 hover:bg-red-500 text-white shadow-lg shadow-red-600/30"
                      : "bg-cyan-600 hover:bg-cyan-500 text-white shadow-lg shadow-cyan-600/20"
                  }`}
                  style={
                    liveMode
                      ? { animation: "glow-red 2s ease-in-out infinite" }
                      : undefined
                  }
                >
                  {liveMode ? (
                    <ZapOff className="w-3.5 h-3.5" />
                  ) : (
                    <Zap className="w-3.5 h-3.5" />
                  )}
                  {liveMode ? "Stop" : "Live"}
                </button>
                <div className="flex-1" />
                <button
                  onClick={stopCamera}
                  className="p-1.5 bg-slate-800 hover:bg-red-900/60 text-slate-400 hover:text-red-300 rounded-lg transition-all border border-slate-700 hover:border-red-800"
                  title="Stop camera"
                >
                  <VideoOff className="w-3.5 h-3.5" />
                </button>
              </>
            )}
          </div>
          {cameraError && (
            <p className="text-[10px] text-red-400 px-2 pb-2 flex items-center gap-1">
              <AlertCircle className="w-3 h-3 shrink-0" />
              {cameraError}
            </p>
          )}
        </div>

        {/* ── Right: Status & Detection Results ── */}
        <div className="flex-1 min-w-0 flex flex-col gap-3">
          {/* Stats bar */}
          <div className="flex items-center gap-2 flex-wrap">
            {fps > 0 && (
              <div
                className={`flex items-center gap-1 bg-slate-900 border border-slate-700/50 px-2.5 py-1.5 rounded-lg text-[11px] font-mono ${fpsColor}`}
              >
                <Activity className="w-3 h-3" />
                {fps} FPS
              </div>
            )}
            {totalPredictions > 0 && (
              <div className="flex items-center gap-1 bg-slate-900 border border-slate-700/50 text-slate-300 px-2.5 py-1.5 rounded-lg text-[11px] font-mono">
                <Hash className="w-3 h-3" />
                {totalPredictions} preds
              </div>
            )}
            {cameraActive && elapsed > 0 && (
              <div className="flex items-center gap-1 bg-slate-900 border border-slate-700/50 text-slate-400 px-2.5 py-1.5 rounded-lg text-[11px] font-mono">
                <Clock className="w-3 h-3" />
                {formatDuration(elapsed)}
              </div>
            )}
            {!cameraActive && !prediction && (
              <p className="text-xs text-slate-400">
                Start the camera to begin detection
              </p>
            )}
          </div>

          {/* Confidence Ring + Class Bars */}
          {prediction ? (
            <motion.div
              className="rounded-xl border border-slate-700/50 bg-slate-900 p-3 shadow-xl flex-1"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              <div className="flex gap-4 items-start">
                {/* Confidence Ring */}
                <div className="flex-shrink-0 flex flex-col items-center">
                  <ConfidenceRing
                    value={prediction.confidence}
                    label={prediction.class_name}
                    size={90}
                  />
                  {liveMode && (
                    <div className="flex items-center gap-1 mt-1.5 text-[9px] text-cyan-400/80 font-medium">
                      <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-pulse" />
                      Real-time
                    </div>
                  )}
                </div>

                {/* Class Bars */}
                <div className="flex-1 min-w-0 space-y-1">
                  <div className="flex items-center justify-between mb-1.5">
                    <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">
                      Class Probabilities
                    </p>
                    {prediction.all_scores.length > 5 && (
                      <span className="text-[9px] text-slate-600">
                        {prediction.all_scores.length} classes
                      </span>
                    )}
                  </div>
                  <div className="space-y-1 max-h-[180px] overflow-y-auto pr-1">
                    {prediction.all_scores.map((s) => {
                      const pct = Math.round(s.score * 100);
                      const isTop = s.class_name === prediction.class_name;
                      const barColor = isTop
                        ? s.score >= 0.8
                          ? "bg-green-500"
                          : s.score >= 0.5
                            ? "bg-yellow-500"
                            : "bg-red-500"
                        : "bg-slate-600";
                      return (
                        <div key={s.class_name}>
                          <div className="flex justify-between text-[10px] mb-0.5">
                            <span
                              className={`font-medium truncate mr-2 ${isTop ? "text-white" : "text-slate-400"}`}
                            >
                              {s.class_name}
                            </span>
                            <span
                              className={`tabular-nums shrink-0 ${isTop ? "text-white font-bold" : "text-slate-500"}`}
                            >
                              {pct}%
                            </span>
                          </div>
                          <div className="h-1 rounded-full bg-slate-800 overflow-hidden">
                            <motion.div
                              className={`h-full rounded-full ${barColor}`}
                              animate={{ width: `${Math.max(pct, 1)}%` }}
                              transition={{ duration: 0.3, ease: "easeOut" }}
                              style={
                                isTop
                                  ? {
                                      boxShadow: `0 0 6px ${
                                        s.score >= 0.8
                                          ? "rgba(34,197,94,0.4)"
                                          : s.score >= 0.5
                                            ? "rgba(234,179,8,0.4)"
                                            : "rgba(239,68,68,0.4)"
                                      }`,
                                    }
                                  : undefined
                              }
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {prediction.all_scores[0] &&
                    prediction.all_scores[0].score < 0.5 && (
                      <p className="text-[9px] text-amber-400/80 flex items-center gap-1 mt-1.5">
                        <AlertCircle className="w-3 h-3 shrink-0" />
                        Low confidence — try better lighting
                      </p>
                    )}
                </div>
              </div>
            </motion.div>
          ) : cameraActive ? (
            <div className="flex-1 rounded-xl border border-slate-700/30 bg-slate-900/50 flex flex-col items-center justify-center gap-2 p-6">
              <div className="w-10 h-10 rounded-full border-2 border-dashed border-slate-700 flex items-center justify-center">
                <Zap className="w-4 h-4 text-slate-600" />
              </div>
              <p className="text-xs text-slate-500 text-center">
                Click <span className="font-semibold text-slate-400">Snap</span> or{" "}
                <span className="font-semibold text-cyan-400">Live</span> to start detecting
              </p>
            </div>
          ) : null}
        </div>
      </div>

      {/* ── Model Info ── */}
      <div className="rounded-lg bg-slate-50 border border-slate-200 p-3">
        <div className="flex items-center justify-between mb-2">
          <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
            Model &middot; {classNames.length} classes
          </p>
          <span className="text-[9px] text-slate-400 font-mono px-1.5 py-0.5 bg-slate-100 rounded">
            {tw}x{th}px &middot; MobileNetV2
          </span>
        </div>
        <div className="flex flex-wrap gap-1">
          {classNames.map((cls) => (
            <span
              key={cls}
              className={`px-2 py-0.5 rounded-md text-[11px] font-medium border transition-all ${
                prediction?.class_name === cls
                  ? "bg-violet-50 border-violet-300 text-violet-700"
                  : "bg-white border-slate-200 text-slate-600"
              }`}
            >
              {cls}
            </span>
          ))}
        </div>
      </div>

      {/* ── CSS Animations ── */}
      <style>{`
        @keyframes scanline {
          0% { top: 0%; }
          100% { top: 100%; }
        }
        @keyframes glow-red {
          0%, 100% { box-shadow: 0 0 8px rgba(239,68,68,0.3); }
          50% { box-shadow: 0 0 20px rgba(239,68,68,0.5); }
        }
      `}</style>
    </div>
  );
};
