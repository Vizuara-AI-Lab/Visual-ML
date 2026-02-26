/**
 * LivePoseTestPanel — run a trained pose model on live camera frames.
 * Uses MediaPipe PoseLandmarker to extract landmarks, sends them to the
 * backend /pose/predict endpoint, and displays the result with skeleton overlay.
 */

import { useRef, useState, useEffect, useCallback } from "react";
import {
  Video,
  VideoOff,
  Camera,
  Zap,
  AlertCircle,
  CheckCircle2,
  Loader2,
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

  const runPrediction = useCallback(async () => {
    if (!lastLandmarksRef.current || predicting) return;
    setPredicting(true);
    try {
      const landmarks = extractLandmarkArray(lastLandmarksRef.current);
      const result = await onPredict(landmarks);
      setPrediction(result);
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

  return (
    <div className="space-y-4">
      {/* Camera + skeleton overlay */}
      <div className="rounded-xl border border-slate-200 bg-slate-900 overflow-hidden">
        <div className="relative aspect-video flex items-center justify-center">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className={`w-full h-full object-cover ${cameraActive ? "opacity-100" : "opacity-0"}`}
          />
          <canvas
            ref={overlayRef}
            className="absolute inset-0 w-full h-full object-cover pointer-events-none"
          />

          {!cameraActive && !modelLoading && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
              <VideoOff className="w-10 h-10 text-slate-500" />
              <p className="text-sm text-slate-400">Camera not started</p>
            </div>
          )}

          {modelLoading && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-slate-900">
              <Loader2 className="w-8 h-8 text-sky-400 animate-spin" />
              <p className="text-sm text-slate-300">Loading pose model...</p>
            </div>
          )}

          {/* Prediction overlay */}
          {prediction && cameraActive && (
            <div className="absolute bottom-3 left-3 right-3 bg-black/70 backdrop-blur-sm rounded-xl p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-lg font-bold text-white">
                  {prediction.class_name}
                </span>
                <span className="text-sm font-mono text-emerald-400">
                  {(prediction.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="space-y-1">
                {prediction.all_scores.slice(0, 5).map((s) => (
                  <div key={s.class_name} className="flex items-center gap-2">
                    <span className="text-[10px] text-slate-300 w-16 truncate">
                      {s.class_name}
                    </span>
                    <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-sky-400 rounded-full transition-all duration-300"
                        style={{ width: `${s.score * 100}%` }}
                      />
                    </div>
                    <span className="text-[10px] text-slate-400 w-10 text-right font-mono">
                      {(s.score * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* No pose warning */}
          {cameraActive && !poseDetected && (
            <div className="absolute top-3 left-3 flex items-center gap-1.5 px-2.5 py-1 bg-amber-500/90 rounded-full">
              <AlertCircle className="w-3 h-3 text-white" />
              <span className="text-xs font-bold text-white">No pose detected</span>
            </div>
          )}
          {cameraActive && poseDetected && (
            <div className="absolute top-3 left-3 flex items-center gap-1.5 px-2.5 py-1 bg-emerald-500/90 rounded-full">
              <CheckCircle2 className="w-3 h-3 text-white" />
              <span className="text-xs font-bold text-white">Pose detected</span>
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2 p-3 bg-slate-800">
          {!cameraActive ? (
            <button
              onClick={startCamera}
              disabled={modelLoading}
              className="flex items-center gap-2 px-4 py-2 bg-sky-600 hover:bg-sky-700 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
            >
              {modelLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Video className="w-4 h-4" />
              )}
              Start Camera
            </button>
          ) : (
            <>
              <button
                onClick={runPrediction}
                disabled={!poseDetected || predicting || autoMode}
                className="flex items-center gap-2 px-4 py-2 bg-white hover:bg-slate-100 text-slate-900 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
              >
                <Camera className="w-4 h-4" />
                {predicting ? "Predicting..." : "Predict"}
              </button>
              <button
                onClick={() => setAutoMode((v) => !v)}
                disabled={!poseDetected}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 ${
                  autoMode
                    ? "bg-sky-500 hover:bg-sky-600 text-white"
                    : "bg-slate-700 hover:bg-slate-600 text-white"
                }`}
              >
                <Zap className="w-4 h-4" />
                {autoMode ? "Live ON" : "Live OFF"}
              </button>
              <button
                onClick={() => { setAutoMode(false); stopCamera(); }}
                className="ml-auto flex items-center gap-2 px-3 py-2 bg-red-900/60 hover:bg-red-900 text-red-300 rounded-lg text-sm transition-colors"
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

      {/* Info */}
      <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
        <p className="text-xs text-slate-500">
          <strong>How it works:</strong> MediaPipe detects 33 body landmarks from
          your camera feed. The landmarks (132 features) are sent to the trained
          model for classification. Press <strong>Live</strong> for automatic
          predictions every second.
        </p>
      </div>
    </div>
  );
};
