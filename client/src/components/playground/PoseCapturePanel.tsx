/**
 * PoseCapturePanel — capture body pose landmarks per class using the device camera.
 * Uses MediaPipe PoseLandmarker to detect 33 body landmarks and stores them as
 * flat 132-float feature vectors (x, y, z, visibility per landmark).
 */

import { useRef, useState, useEffect, useCallback } from "react";
import {
  Camera,
  Video,
  VideoOff,
  Trash2,
  CheckCircle2,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  Play,
  Zap,
  Loader2,
} from "lucide-react";
import {
  getPoseLandmarker,
  extractLandmarkArray,
  drawPoseSkeleton,
  N_FEATURES,
} from "../../lib/poseLandmarker";
import type { PoseLandmarker as PoseLandmarkerType } from "@mediapipe/tasks-vision";

interface CapturedPose {
  landmarks: number[]; // flat 132-float array
  dataUrl: string; // screenshot preview
}

interface ClassPoses {
  [className: string]: CapturedPose[];
}

interface PoseCaptureProps {
  config: Record<string, unknown>;
  onDatasetReady: (datasetPayload: {
    class_names: string[];
    landmarks_per_class: Record<string, number[][]>;
  }) => void;
  isSubmitting?: boolean;
}

export const PoseCapturePanel = ({
  config,
  onDatasetReady,
  isSubmitting,
}: PoseCaptureProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<PoseLandmarkerType | null>(null);
  const animRef = useRef<number | null>(null);
  const lastLandmarksRef = useRef<
    { x: number; y: number; z: number; visibility?: number }[] | null
  >(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [modelLoading, setModelLoading] = useState(false);
  const [poseDetected, setPoseDetected] = useState(false);
  const [classPoses, setClassPoses] = useState<ClassPoses>({});
  const [activeClass, setActiveClass] = useState<string>("");
  const [countdown, setCountdown] = useState<number | null>(null);
  const [burstActive, setBurstActive] = useState(false);
  const burstRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const classNamesRaw = (config.class_names as string) || "";
  const classNames = classNamesRaw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const samplesPerClass = Number(config.samples_per_class) || 30;
  const minRequired = Math.min(5, samplesPerClass);

  // Keep classPoses in sync when class list changes
  useEffect(() => {
    setClassPoses((prev) => {
      const updated: ClassPoses = {};
      classNames.forEach((cls) => {
        updated[cls] = prev[cls] || [];
      });
      return updated;
    });
    if (classNames.length > 0 && !activeClass) {
      setActiveClass(classNames[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [classNamesRaw]);

  // --- MediaPipe detection loop ---
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

      // Ensure overlay matches video display size
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;

      const result = landmarker.detectForVideo(video, performance.now());

      if (result.landmarks && result.landmarks.length > 0) {
        const lm = result.landmarks[0];
        lastLandmarksRef.current = lm;
        setPoseDetected(true);
        drawPoseSkeleton(ctx, lm, overlay.width, overlay.height);
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
      // Load MediaPipe model (cached after first load)
      const landmarker = await getPoseLandmarker();
      landmarkerRef.current = landmarker;

      // Start camera
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
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
    setPoseDetected(false);
    lastLandmarksRef.current = null;
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  // --- Capture current pose ---
  const capturePose = useCallback(() => {
    if (!lastLandmarksRef.current || !activeClass) return;

    const landmarks = extractLandmarkArray(lastLandmarksRef.current);
    if (landmarks.length !== N_FEATURES) return;

    // Capture a screenshot for preview
    const video = videoRef.current;
    const overlay = overlayRef.current;
    if (!video) return;

    const preview = document.createElement("canvas");
    preview.width = 112;
    preview.height = 112;
    const pctx = preview.getContext("2d")!;
    // Draw video frame
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const side = Math.min(vw, vh);
    const sx = (vw - side) / 2;
    const sy = (vh - side) / 2;
    pctx.drawImage(video, sx, sy, side, side, 0, 0, 112, 112);
    // Draw skeleton overlay on top
    if (overlay) {
      pctx.drawImage(overlay, sx, sy, side, side, 0, 0, 112, 112);
    }

    setClassPoses((prev) => ({
      ...prev,
      [activeClass]: [
        ...(prev[activeClass] || []),
        { landmarks, dataUrl: preview.toDataURL("image/png") },
      ],
    }));
  }, [activeClass]);

  // 3-2-1 countdown then capture
  const startCountdown = useCallback(() => {
    setCountdown(3);
    let count = 3;
    const interval = setInterval(() => {
      count -= 1;
      if (count === 0) {
        clearInterval(interval);
        setCountdown(null);
        capturePose();
      } else {
        setCountdown(count);
      }
    }, 1000);
  }, [capturePose]);

  // Burst capture — one pose every 500ms
  const stopBurst = useCallback(() => {
    if (burstRef.current) {
      clearInterval(burstRef.current);
      burstRef.current = null;
    }
    setBurstActive(false);
  }, []);

  const startBurst = useCallback(() => {
    if (!cameraActive || !activeClass) return;
    setBurstActive(true);
    capturePose();
    burstRef.current = setInterval(() => {
      capturePose();
    }, 500);
  }, [cameraActive, activeClass, capturePose]);

  useEffect(() => {
    return () => stopBurst();
  }, [activeClass, cameraActive, stopBurst]);

  const deletePose = (cls: string, idx: number) => {
    setClassPoses((prev) => ({
      ...prev,
      [cls]: prev[cls].filter((_, i) => i !== idx),
    }));
  };

  const totalPoses = Object.values(classPoses).reduce(
    (s, arr) => s + arr.length,
    0,
  );
  const deficientClasses = classNames.filter(
    (cls) => (classPoses[cls]?.length || 0) < minRequired,
  );
  const allClassesMeetMinimum = deficientClasses.length === 0;

  const handleBuildDataset = () => {
    const payload: Record<string, number[][]> = {};
    classNames.forEach((cls) => {
      payload[cls] = (classPoses[cls] || []).map((p) => p.landmarks);
    });
    onDatasetReady({
      class_names: classNames,
      landmarks_per_class: payload,
    });
  };

  if (classNames.length === 0) {
    return (
      <div className="rounded-xl border border-amber-200 bg-amber-50/60 p-4 flex items-start gap-3">
        <AlertCircle className="w-4 h-4 text-amber-500 mt-0.5 shrink-0" />
        <div>
          <p className="text-sm font-semibold text-amber-800">
            Set Pose Labels First
          </p>
          <p className="text-xs text-amber-700 mt-0.5">
            Enter comma-separated pose names above (e.g.{" "}
            <code>standing, sitting, waving</code>), then the camera will appear
            here.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Camera preview with skeleton overlay */}
      <div className="rounded-xl border border-slate-200 bg-slate-900 overflow-hidden">
        <div className="relative aspect-video flex items-center justify-center">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className={`w-full h-full object-cover ${cameraActive ? "opacity-100" : "opacity-0"}`}
          />
          {/* Skeleton overlay canvas */}
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
              <p className="text-sm text-slate-300">
                Loading pose detection model...
              </p>
            </div>
          )}

          {/* No pose warning */}
          {cameraActive && !poseDetected && (
            <div className="absolute top-3 left-3 flex items-center gap-1.5 px-2.5 py-1 bg-amber-500/90 rounded-full">
              <AlertCircle className="w-3 h-3 text-white" />
              <span className="text-xs font-bold text-white">
                No pose detected
              </span>
            </div>
          )}

          {/* Pose detected indicator */}
          {cameraActive && poseDetected && (
            <div className="absolute top-3 left-3 flex items-center gap-1.5 px-2.5 py-1 bg-emerald-500/90 rounded-full">
              <CheckCircle2 className="w-3 h-3 text-white" />
              <span className="text-xs font-bold text-white">
                Pose detected
              </span>
            </div>
          )}

          {/* Countdown overlay */}
          {countdown !== null && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/40">
              <span className="text-7xl font-black text-white drop-shadow-2xl">
                {countdown}
              </span>
            </div>
          )}

          {/* Burst indicator */}
          {burstActive && (
            <div className="absolute top-3 right-3 flex items-center gap-1.5 px-2.5 py-1 bg-orange-500 rounded-full animate-pulse">
              <Zap className="w-3 h-3 text-white" />
              <span className="text-xs font-bold text-white">BURST</span>
            </div>
          )}
        </div>

        {/* Camera controls */}
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
                onClick={capturePose}
                disabled={
                  !activeClass ||
                  !poseDetected ||
                  countdown !== null ||
                  burstActive
                }
                className="flex items-center gap-2 px-4 py-2 bg-white hover:bg-slate-100 text-slate-900 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
              >
                <Camera className="w-4 h-4" />
                Snap
              </button>
              <button
                onClick={startCountdown}
                disabled={
                  !activeClass ||
                  !poseDetected ||
                  countdown !== null ||
                  burstActive
                }
                className="flex items-center gap-2 px-3 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
              >
                3-2-1
              </button>
              <button
                onClick={burstActive ? stopBurst : startBurst}
                disabled={!activeClass || !poseDetected || countdown !== null}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 ${
                  burstActive
                    ? "bg-orange-500 hover:bg-orange-600 text-white"
                    : "bg-slate-700 hover:bg-slate-600 text-white"
                }`}
              >
                <Zap className="w-4 h-4" />
                {burstActive ? "Stop" : "Burst"}
              </button>
              <button
                onClick={() => {
                  stopBurst();
                  stopCamera();
                }}
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

      {/* Class tabs */}
      <div className="flex gap-1 overflow-x-auto pb-1">
        {classNames.map((cls) => {
          const count = classPoses[cls]?.length || 0;
          const meetsMin = count >= minRequired;
          const isFull = count >= samplesPerClass;
          return (
            <button
              key={cls}
              onClick={() => {
                stopBurst();
                setActiveClass(cls);
              }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                activeClass === cls
                  ? "bg-sky-600 text-white shadow"
                  : meetsMin
                    ? "bg-slate-100 text-slate-600 hover:bg-slate-200"
                    : "bg-amber-50 text-amber-700 hover:bg-amber-100 border border-amber-200"
              }`}
            >
              {isFull && <CheckCircle2 className="w-3 h-3 text-emerald-400" />}
              {cls}
              <span
                className={`px-1.5 py-0.5 rounded-full text-[10px] font-bold ${
                  activeClass === cls
                    ? "bg-white/20"
                    : meetsMin
                      ? "bg-slate-200"
                      : "bg-amber-200 text-amber-800"
                }`}
              >
                {count}/{samplesPerClass}
              </span>
            </button>
          );
        })}
      </div>

      {/* Captured poses for active class */}
      {activeClass && (
        <div>
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm font-semibold text-slate-700">
              &ldquo;{activeClass}&rdquo; &mdash;{" "}
              {classPoses[activeClass]?.length || 0} poses
            </p>
            <div className="flex gap-1">
              {classNames.indexOf(activeClass) > 0 && (
                <button
                  onClick={() => {
                    stopBurst();
                    setActiveClass(
                      classNames[classNames.indexOf(activeClass) - 1],
                    );
                  }}
                  className="p-1 hover:bg-slate-100 rounded"
                >
                  <ChevronLeft className="w-4 h-4 text-slate-400" />
                </button>
              )}
              {classNames.indexOf(activeClass) < classNames.length - 1 && (
                <button
                  onClick={() => {
                    stopBurst();
                    setActiveClass(
                      classNames[classNames.indexOf(activeClass) + 1],
                    );
                  }}
                  className="p-1 hover:bg-slate-100 rounded"
                >
                  <ChevronRight className="w-4 h-4 text-slate-400" />
                </button>
              )}
            </div>
          </div>

          {(classPoses[activeClass]?.length || 0) === 0 ? (
            <div className="rounded-lg border-2 border-dashed border-slate-200 bg-slate-50 py-8 text-center">
              <Camera className="w-8 h-8 text-slate-300 mx-auto mb-2" />
              <p className="text-xs text-slate-400">
                No poses yet — stand in front of camera & press Snap or Burst
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-5 gap-1.5">
              {(classPoses[activeClass] || []).map((pose, i) => (
                <div
                  key={i}
                  className="relative group rounded-lg overflow-hidden border border-slate-200 bg-slate-900 aspect-square"
                >
                  <img
                    src={pose.dataUrl}
                    alt={`${activeClass} ${i}`}
                    className="w-full h-full object-cover"
                  />
                  <button
                    onClick={() => deletePose(activeClass, i)}
                    className="absolute top-0.5 right-0.5 p-0.5 bg-red-600 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <Trash2 className="w-2.5 h-2.5 text-white" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Summary + Build button */}
      <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-3">
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="bg-sky-50 rounded-lg py-2">
            <div className="text-lg font-bold text-sky-700">
              {classNames.length}
            </div>
            <div className="text-[10px] font-bold text-sky-400 uppercase tracking-wider">
              Poses
            </div>
          </div>
          <div className="bg-slate-50 rounded-lg py-2">
            <div className="text-lg font-bold text-slate-700">{totalPoses}</div>
            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
              Total Captures
            </div>
          </div>
          <div className="bg-slate-50 rounded-lg py-2">
            <div className="text-lg font-bold text-slate-700">132</div>
            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
              Features
            </div>
          </div>
        </div>

        {!allClassesMeetMinimum && (
          <div className="text-xs text-amber-600 flex items-start gap-1.5">
            <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
            <div>
              <p className="font-medium">
                Need at least {minRequired} captures per pose:
              </p>
              <ul className="mt-0.5 space-y-0.5">
                {deficientClasses.map((cls) => (
                  <li key={cls}>
                    <strong>{cls}</strong> &mdash;{" "}
                    {classPoses[cls]?.length || 0}/{minRequired} captures
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        <button
          onClick={handleBuildDataset}
          disabled={!allClassesMeetMinimum || isSubmitting}
          className="w-full flex items-center justify-center gap-2 py-2.5 bg-sky-600 hover:bg-sky-700 disabled:bg-slate-200 disabled:text-slate-400 text-white rounded-lg text-sm font-semibold transition-colors"
        >
          <Play className="w-4 h-4" />
          {isSubmitting ? "Building Dataset..." : "Build Dataset & Continue"}
        </button>
      </div>
    </div>
  );
};
