/**
 * CameraCapturePanel — capture training images per class using the device camera.
 * Converts captured frames to grayscale pixel arrays matching the image pipeline format.
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
} from "lucide-react";

interface CapturedImage {
  dataUrl: string;   // high-res preview
  pixels: number[];  // flattened grayscale pixels at target size
}

interface ClassImages {
  [className: string]: CapturedImage[];
}

interface CameraCaptureProps {
  config: Record<string, unknown>;
  onDatasetReady: (datasetPayload: {
    class_names: string[];
    target_size: string;
    images_per_class: Record<string, number[][]>; // className -> list of pixel arrays
  }) => void;
  isSubmitting?: boolean;
}

// Parse "WxH" string → [width, height]
function parseSize(sizeStr: string): [number, number] {
  const parts = (sizeStr || "28x28").toLowerCase().split("x");
  const w = parseInt(parts[0]) || 28;
  const h = parseInt(parts[1]) || 28;
  return [w, h];
}

const PREVIEW_SIZE = 112;

// Render a video frame → ML-resolution pixel array + high-res preview
function frameToGrayscalePixels(
  video: HTMLVideoElement,
  width: number,
  height: number,
): { pixels: number[]; dataUrl: string } {
  // Common crop region from video
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const side = Math.min(vw, vh);
  const sx = (vw - side) / 2;
  const sy = (vh - side) / 2;

  // 1. ML-resolution canvas for pixel array
  const mlCanvas = document.createElement("canvas");
  mlCanvas.width = width;
  mlCanvas.height = height;
  const mlCtx = mlCanvas.getContext("2d")!;
  mlCtx.fillStyle = "#000";
  mlCtx.fillRect(0, 0, width, height);
  mlCtx.drawImage(video, sx, sy, side, side, 0, 0, width, height);

  const imageData = mlCtx.getImageData(0, 0, width, height);
  const data = imageData.data;
  const pixels: number[] = [];
  for (let i = 0; i < data.length; i += 4) {
    const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    pixels.push(gray);
  }

  // 2. High-res preview from ORIGINAL video (not from tiny ML canvas)
  const previewCanvas = document.createElement("canvas");
  previewCanvas.width = PREVIEW_SIZE;
  previewCanvas.height = PREVIEW_SIZE;
  const pctx = previewCanvas.getContext("2d")!;
  pctx.imageSmoothingEnabled = true;
  pctx.imageSmoothingQuality = "high";
  pctx.drawImage(video, sx, sy, side, side, 0, 0, PREVIEW_SIZE, PREVIEW_SIZE);
  // Convert preview to grayscale for visual consistency
  const previewData = pctx.getImageData(0, 0, PREVIEW_SIZE, PREVIEW_SIZE);
  const pd = previewData.data;
  for (let i = 0; i < pd.length; i += 4) {
    const gray = Math.round(0.299 * pd[i] + 0.587 * pd[i + 1] + 0.114 * pd[i + 2]);
    pd[i] = gray;
    pd[i + 1] = gray;
    pd[i + 2] = gray;
  }
  pctx.putImageData(previewData, 0, 0);

  return { pixels, dataUrl: previewCanvas.toDataURL("image/png") };
}

export const CameraCapturePanel = ({
  config,
  onDatasetReady,
  isSubmitting,
}: CameraCaptureProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [classImages, setClassImages] = useState<ClassImages>({});
  const [activeClass, setActiveClass] = useState<string>("");
  const [countdown, setCountdown] = useState<number | null>(null);
  const [burstActive, setBurstActive] = useState(false);
  const burstRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const classNamesRaw = (config.class_names as string) || "";
  const classNames = classNamesRaw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const targetSize = (config.target_size as string) || "28x28";
  const [tw, th] = parseSize(targetSize);
  const imagesPerClass = Number(config.images_per_class) || 20;
  const minRequired = Math.min(5, imagesPerClass);

  // Keep classImages in sync when class list changes
  useEffect(() => {
    setClassImages((prev) => {
      const updated: ClassImages = {};
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
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const capturePhoto = useCallback(() => {
    if (!videoRef.current || !cameraActive || !activeClass) return;
    const { pixels, dataUrl } = frameToGrayscalePixels(videoRef.current, tw, th);
    setClassImages((prev) => ({
      ...prev,
      [activeClass]: [...(prev[activeClass] || []), { pixels, dataUrl }],
    }));
  }, [cameraActive, activeClass, tw, th]);

  // 3-2-1 countdown then capture
  const startCountdown = useCallback(() => {
    setCountdown(3);
    let count = 3;
    const interval = setInterval(() => {
      count -= 1;
      if (count === 0) {
        clearInterval(interval);
        setCountdown(null);
        capturePhoto();
      } else {
        setCountdown(count);
      }
    }, 1000);
  }, [capturePhoto]);

  // Burst capture — one photo every 500ms
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
    capturePhoto();
    burstRef.current = setInterval(() => {
      capturePhoto();
    }, 500);
  }, [cameraActive, activeClass, capturePhoto]);

  // Auto-stop burst when class/camera changes or component unmounts
  useEffect(() => {
    return () => stopBurst();
  }, [activeClass, cameraActive, stopBurst]);

  const deleteImage = (cls: string, idx: number) => {
    setClassImages((prev) => ({
      ...prev,
      [cls]: prev[cls].filter((_, i) => i !== idx),
    }));
  };

  const totalImages = Object.values(classImages).reduce((s, arr) => s + arr.length, 0);
  const deficientClasses = classNames.filter(
    (cls) => (classImages[cls]?.length || 0) < minRequired,
  );
  const allClassesMeetMinimum = deficientClasses.length === 0;

  const handleBuildDataset = () => {
    const payload: Record<string, number[][]> = {};
    classNames.forEach((cls) => {
      payload[cls] = (classImages[cls] || []).map((img) => img.pixels);
    });
    onDatasetReady({
      class_names: classNames,
      target_size: targetSize,
      images_per_class: payload,
    });
  };

  if (classNames.length === 0) {
    return (
      <div className="rounded-xl border border-amber-200 bg-amber-50/60 p-4 flex items-start gap-3">
        <AlertCircle className="w-4 h-4 text-amber-500 mt-0.5 shrink-0" />
        <div>
          <p className="text-sm font-semibold text-amber-800">Set Class Labels First</p>
          <p className="text-xs text-amber-700 mt-0.5">
            Enter comma-separated class names above (e.g. <code>rock,paper,scissors</code>), then the camera will appear here.
          </p>
        </div>
      </div>
    );
  }

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

          {/* Square crop guide */}
          {cameraActive && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="border-2 border-white/40 rounded-lg"
                style={{ width: "60%", aspectRatio: "1/1" }} />
            </div>
          )}
        </div>

        {/* Camera controls */}
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
                onClick={capturePhoto}
                disabled={!activeClass || countdown !== null || burstActive}
                className="flex items-center gap-2 px-4 py-2 bg-white hover:bg-slate-100 text-slate-900 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
              >
                <Camera className="w-4 h-4" />
                Snap
              </button>
              <button
                onClick={startCountdown}
                disabled={!activeClass || countdown !== null || burstActive}
                className="flex items-center gap-2 px-3 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
              >
                3-2-1
              </button>
              <button
                onClick={burstActive ? stopBurst : startBurst}
                disabled={!activeClass || countdown !== null}
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
                onClick={() => { stopBurst(); stopCamera(); }}
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
          const count = classImages[cls]?.length || 0;
          const meetsMin = count >= minRequired;
          const isFull = count >= imagesPerClass;
          return (
            <button
              key={cls}
              onClick={() => { stopBurst(); setActiveClass(cls); }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                activeClass === cls
                  ? "bg-violet-600 text-white shadow"
                  : meetsMin
                    ? "bg-slate-100 text-slate-600 hover:bg-slate-200"
                    : "bg-amber-50 text-amber-700 hover:bg-amber-100 border border-amber-200"
              }`}
            >
              {isFull && <CheckCircle2 className="w-3 h-3 text-emerald-400" />}
              {cls}
              <span className={`px-1.5 py-0.5 rounded-full text-[10px] font-bold ${
                activeClass === cls ? "bg-white/20" : meetsMin ? "bg-slate-200" : "bg-amber-200 text-amber-800"
              }`}>
                {count}/{imagesPerClass}
              </span>
            </button>
          );
        })}
      </div>

      {/* Captured images for active class */}
      {activeClass && (
        <div>
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm font-semibold text-slate-700">
              &ldquo;{activeClass}&rdquo; — {classImages[activeClass]?.length || 0} photos
            </p>
            <div className="flex gap-1">
              {classNames.indexOf(activeClass) > 0 && (
                <button
                  onClick={() => { stopBurst(); setActiveClass(classNames[classNames.indexOf(activeClass) - 1]); }}
                  className="p-1 hover:bg-slate-100 rounded"
                >
                  <ChevronLeft className="w-4 h-4 text-slate-400" />
                </button>
              )}
              {classNames.indexOf(activeClass) < classNames.length - 1 && (
                <button
                  onClick={() => { stopBurst(); setActiveClass(classNames[classNames.indexOf(activeClass) + 1]); }}
                  className="p-1 hover:bg-slate-100 rounded"
                >
                  <ChevronRight className="w-4 h-4 text-slate-400" />
                </button>
              )}
            </div>
          </div>

          {(classImages[activeClass]?.length || 0) === 0 ? (
            <div className="rounded-lg border-2 border-dashed border-slate-200 bg-slate-50 py-8 text-center">
              <Camera className="w-8 h-8 text-slate-300 mx-auto mb-2" />
              <p className="text-xs text-slate-400">
                No photos yet — point camera & press Snap or Burst
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-5 gap-1.5">
              {(classImages[activeClass] || []).map((img, i) => (
                <div key={i} className="relative group rounded-lg overflow-hidden border border-slate-200 bg-slate-900 aspect-square">
                  <img
                    src={img.dataUrl}
                    alt={`${activeClass} ${i}`}
                    className="w-full h-full object-cover"
                  />
                  <button
                    onClick={() => deleteImage(activeClass, i)}
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
          <div className="bg-violet-50 rounded-lg py-2">
            <div className="text-lg font-bold text-violet-700">{classNames.length}</div>
            <div className="text-[10px] font-bold text-violet-400 uppercase tracking-wider">Classes</div>
          </div>
          <div className="bg-slate-50 rounded-lg py-2">
            <div className="text-lg font-bold text-slate-700">{totalImages}</div>
            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Total Photos</div>
          </div>
          <div className="bg-slate-50 rounded-lg py-2">
            <div className="text-lg font-bold text-slate-700">{targetSize}</div>
            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Pixel Size</div>
          </div>
        </div>

        {!allClassesMeetMinimum && (
          <div className="text-xs text-amber-600 flex items-start gap-1.5">
            <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
            <div>
              <p className="font-medium">Need at least {minRequired} photos per class:</p>
              <ul className="mt-0.5 space-y-0.5">
                {deficientClasses.map((cls) => (
                  <li key={cls}>
                    <strong>{cls}</strong> — {classImages[cls]?.length || 0}/{minRequired} photos
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        <button
          onClick={handleBuildDataset}
          disabled={!allClassesMeetMinimum || isSubmitting}
          className="w-full flex items-center justify-center gap-2 py-2.5 bg-violet-600 hover:bg-violet-700 disabled:bg-slate-200 disabled:text-slate-400 text-white rounded-lg text-sm font-semibold transition-colors"
        >
          <Play className="w-4 h-4" />
          {isSubmitting ? "Building Dataset…" : "Build Dataset & Continue"}
        </button>
      </div>
    </div>
  );
};
