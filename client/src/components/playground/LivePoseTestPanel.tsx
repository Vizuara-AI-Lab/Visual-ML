/**
 * LivePoseTestPanel — run a trained pose model on live camera frames.
 * Uses MediaPipe PoseLandmarker to extract landmarks, sends them to the
 * backend /pose/predict endpoint, and displays the result with skeleton overlay.
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
  CheckCircle2,
  Loader2,
  Activity,
  Hash,
  Clock,
} from "lucide-react";
import {
  getPoseLandmarker,
  extractLandmarkArray,
  drawPoseSkeleton,
} from "../../lib/poseLandmarker";
import type { PoseLandmarker as PoseLandmarkerType } from "@mediapipe/tasks-vision";

interface Prediction {
  class_name: string;
  confidence: number;
  all_scores: { class_name: string; score: number }[];
}

interface LivePoseTestPanelProps {
  modelId: string;
  modelPath: string;
  classNames: string[];
  onPredict: (landmarks: number[]) => Promise<Prediction>;
}

export const LivePoseTestPanel = ({
  modelId,
  classNames,
  onPredict,
}: LivePoseTestPanelProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<PoseLandmarkerType | null>(null);
  const animRef = useRef<number | null>(null);
  const autoRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastLandmarksRef = useRef<
    { x: number; y: number; z: number; visibility?: number }[] | null
  >(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [modelLoading, setModelLoading] = useState(false);
  const [poseDetected, setPoseDetected] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [autoMode, setAutoMode] = useState(false);
  const [totalPredictions, setTotalPredictions] = useState(0);
  const [sessionStart] = useState(Date.now());
  const [elapsed, setElapsed] = useState(0);

  // Detection loop
  const runDetectionLoop = useCallback(() => {
    const detect = () => {
      const video = videoRef.current;
      const overlay = overlayRef.current;
      const landmarker = landmarkerRef.current;
      if (!video || !overlay || !landmarker || video.readyState < 2) {
        animRef.current = requestAnimationFrame(detect);
        return;
      }
      const ctx = overlay.getContext("2d");
      if (!ctx) {
        animRef.current = requestAnimationFrame(detect);
        return;
      }
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;

      const result = landmarker.detectForVideo(video, performance.now());
      if (result.landmarks && result.landmarks.length > 0) {
        lastLandmarksRef.current = result.landmarks[0];
        setPoseDetected(true);
        drawPoseSkeleton(ctx, result.landmarks[0], overlay.width, overlay.height);
      } else {
        lastLandmarksRef.current = null;
        setPoseDetected(false);
        ctx.clearRect(0, 0, overlay.width, overlay.height);
      }
      animRef.current = requestAnimationFrame(detect);
    };
    animRef.current = requestAnimationFrame(detect);
  }, []);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    setModelLoading(true);
    try {
      const landmarker = await getPoseLandmarker();
      landmarkerRef.current = landmarker;

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);
      setModelLoading(false);
      runDetectionLoop();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setCameraError(`Failed to start: ${msg}`);
      setModelLoading(false);
    }
  }, [runDetectionLoop]);

  const stopCamera = useCallback(() => {
    if (animRef.current) {
      cancelAnimationFrame(animRef.current);
      animRef.current = null;
    }
    if (autoRef.current) {
      clearInterval(autoRef.current);
      autoRef.current = null;
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
    setPoseDetected(false);
    setAutoMode(false);
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  // Session timer
  useEffect(() => {
    if (!cameraActive) return;
    const timer = setInterval(() => setElapsed(Date.now() - sessionStart), 1000);
    return () => clearInterval(timer);
  }, [cameraActive, sessionStart]);

  const runPrediction = useCallback(async () => {
    if (!lastLandmarksRef.current || predicting) return;
    setPredicting(true);
    try {
      const landmarks = extractLandmarkArray(lastLandmarksRef.current);
      const result = await onPredict(landmarks);
      setPrediction(result);
      setTotalPredictions((n) => n + 1);
    } catch {
      // silently ignore mid-stream errors
    } finally {
      setPredicting(false);
    }
  }, [onPredict, predicting]);

  // Auto-predict loop
  useEffect(() => {
    if (autoMode && cameraActive) {
      runPrediction();
      autoRef.current = setInterval(runPrediction, 1000);
    } else if (autoRef.current) {
      clearInterval(autoRef.current);
      autoRef.current = null;
    }
    return () => {
      if (autoRef.current) {
        clearInterval(autoRef.current);
        autoRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoMode, cameraActive]);

  const confidencePct = prediction
    ? Math.round(prediction.confidence * 100)
    : 0;

  const formatDuration = (ms: number) => {
    const s = Math.floor(ms / 1000);
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  return (
    <div className="space-y-3">
      {/* ── Two-Column Layout: Camera (left) + Results (right) ── */}
      <div className="flex gap-3">
        {/* ── Left: Camera + Skeleton ── */}
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
            <canvas
              ref={overlayRef}
              className="absolute inset-0 w-full h-full object-cover pointer-events-none"
            />

            {/* Idle state */}
            {!cameraActive && !modelLoading && (
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

            {/* Loading state */}
            {modelLoading && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-slate-900">
                <Loader2 className="w-7 h-7 text-sky-400 animate-spin" />
                <p className="text-[11px] text-slate-300">Loading pose model...</p>
              </div>
            )}

            {/* HUD corners */}
            {cameraActive && (
              <>
                <div className={`absolute top-3 left-3 w-7 h-7 border-t-2 border-l-2 border-sky-400/50 rounded-tl transition-opacity ${autoMode ? "opacity-100 animate-pulse" : "opacity-40"}`} />
                <div className={`absolute top-3 right-3 w-7 h-7 border-t-2 border-r-2 border-sky-400/50 rounded-tr transition-opacity ${autoMode ? "opacity-100 animate-pulse" : "opacity-40"}`} />
                <div className={`absolute bottom-3 left-3 w-7 h-7 border-b-2 border-l-2 border-sky-400/50 rounded-bl transition-opacity ${autoMode ? "opacity-100 animate-pulse" : "opacity-40"}`} />
                <div className={`absolute bottom-3 right-3 w-7 h-7 border-b-2 border-r-2 border-sky-400/50 rounded-br transition-opacity ${autoMode ? "opacity-100 animate-pulse" : "opacity-40"}`} />
              </>
            )}

            {/* Pose status badge */}
            {cameraActive && !poseDetected && (
              <div className="absolute top-2 left-2 flex items-center gap-1 bg-amber-500/90 text-white px-2 py-0.5 rounded text-[9px] font-bold">
                <AlertCircle className="w-3 h-3" />
                No pose
              </div>
            )}
            {cameraActive && poseDetected && (
              <div className="absolute top-2 left-2 flex items-center gap-1 bg-emerald-500/90 text-white px-2 py-0.5 rounded text-[9px] font-bold">
                <CheckCircle2 className="w-3 h-3" />
                Pose OK
              </div>
            )}

            {/* Auto mode badge */}
            {cameraActive && autoMode && (
              <div className="absolute top-2 right-2 flex items-center gap-1 bg-red-600/90 text-white px-2 py-0.5 rounded text-[9px] font-bold uppercase tracking-widest">
                <span className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
                Live
              </div>
            )}

            {/* Compact prediction on video */}
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
                    <div className="w-2.5 h-2.5 border-2 border-sky-400 border-t-transparent rounded-full animate-spin shrink-0" />
                  )}
                </div>
              </div>
            )}
          </div>

          {/* ── Controls ── */}
          <div className="flex items-center gap-1.5 px-2 py-2 bg-slate-900 border-t border-slate-800">
            {!cameraActive ? (
              <button
                onClick={startCamera}
                disabled={modelLoading}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-sky-600 hover:bg-sky-500 text-white rounded-lg text-xs font-semibold transition-all disabled:opacity-50"
              >
                {modelLoading ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Video className="w-3.5 h-3.5" />
                )}
                Start
              </button>
            ) : (
              <>
                <button
                  onClick={runPrediction}
                  disabled={!poseDetected || predicting || autoMode}
                  className="flex items-center gap-1 px-2.5 py-1.5 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-xs font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <Camera className="w-3.5 h-3.5" />
                  Snap
                </button>
                <button
                  onClick={() => setAutoMode((v) => !v)}
                  disabled={!poseDetected}
                  className={`flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-semibold transition-all disabled:opacity-40 ${
                    autoMode
                      ? "bg-red-600 hover:bg-red-500 text-white shadow-lg shadow-red-600/30"
                      : "bg-sky-600 hover:bg-sky-500 text-white shadow-lg shadow-sky-600/20"
                  }`}
                >
                  {autoMode ? (
                    <ZapOff className="w-3.5 h-3.5" />
                  ) : (
                    <Zap className="w-3.5 h-3.5" />
                  )}
                  {autoMode ? "Stop" : "Live"}
                </button>
                <div className="flex-1" />
                <button
                  onClick={() => { setAutoMode(false); stopCamera(); }}
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
            {cameraActive && poseDetected && (
              <div className="flex items-center gap-1 bg-emerald-900/40 border border-emerald-700/50 text-emerald-400 px-2.5 py-1.5 rounded-lg text-[11px] font-medium">
                <Activity className="w-3 h-3" />
                Tracking
              </div>
            )}
            {!cameraActive && !prediction && (
              <p className="text-xs text-slate-400">
                Start the camera to begin pose detection
              </p>
            )}
          </div>

          {/* Detection results */}
          {prediction ? (
            <motion.div
              className="rounded-xl border border-slate-700/50 bg-slate-900 p-3 shadow-xl flex-1"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              {/* Detected class + confidence */}
              <div className="flex items-center gap-3 mb-3 pb-3 border-b border-slate-800">
                <div className="flex-shrink-0 w-12 h-12 rounded-xl bg-slate-800 flex items-center justify-center">
                  <motion.span
                    className={`text-lg font-black tabular-nums ${
                      confidencePct >= 80
                        ? "text-green-400"
                        : confidencePct >= 50
                          ? "text-yellow-400"
                          : "text-red-400"
                    }`}
                    key={confidencePct}
                    initial={{ scale: 1.2, opacity: 0.7 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.2 }}
                  >
                    {confidencePct}%
                  </motion.span>
                </div>
                <div className="flex-1 min-w-0">
                  <motion.p
                    className="text-white font-bold text-base truncate"
                    key={prediction.class_name}
                    initial={{ opacity: 0, x: -6 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.15 }}
                  >
                    {prediction.class_name}
                  </motion.p>
                  <p className="text-[10px] text-slate-500">Detected pose</p>
                </div>
                {autoMode && (
                  <div className="flex items-center gap-1 text-[9px] text-sky-400/80 font-medium">
                    <span className="w-1.5 h-1.5 bg-sky-400 rounded-full animate-pulse" />
                    Real-time
                  </div>
                )}
              </div>

              {/* Class bars */}
              <div className="space-y-1 max-h-[160px] overflow-y-auto pr-1">
                <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1.5">
                  Class Probabilities
                </p>
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
                  <p className="text-[9px] text-amber-400/80 flex items-center gap-1 mt-2">
                    <AlertCircle className="w-3 h-3 shrink-0" />
                    Low confidence — adjust your pose
                  </p>
                )}
            </motion.div>
          ) : cameraActive ? (
            <div className="flex-1 rounded-xl border border-slate-700/30 bg-slate-900/50 flex flex-col items-center justify-center gap-2 p-6">
              <div className="w-10 h-10 rounded-full border-2 border-dashed border-slate-700 flex items-center justify-center">
                <Zap className="w-4 h-4 text-slate-600" />
              </div>
              <p className="text-xs text-slate-500 text-center">
                {poseDetected ? (
                  <>Click <span className="font-semibold text-slate-400">Snap</span> or <span className="font-semibold text-sky-400">Live</span> to classify</>
                ) : (
                  <>Stand in view so MediaPipe can detect your pose</>
                )}
              </p>
            </div>
          ) : null}
        </div>
      </div>

      {/* ── Info ── */}
      <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
        <div className="flex items-center justify-between mb-1">
          <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
            Pose Detection &middot; {classNames.length} classes
          </p>
          <span className="text-[9px] text-slate-400 font-mono px-1.5 py-0.5 bg-slate-100 rounded">
            33 landmarks &middot; MediaPipe
          </span>
        </div>
        <div className="flex flex-wrap gap-1">
          {classNames.map((cls) => (
            <span
              key={cls}
              className={`px-2 py-0.5 rounded-md text-[11px] font-medium border transition-all ${
                prediction?.class_name === cls
                  ? "bg-sky-50 border-sky-300 text-sky-700"
                  : "bg-white border-slate-200 text-slate-600"
              }`}
            >
              {cls}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};
