/**
 * Shared MediaPipe PoseLandmarker utility.
 * Singleton loader — caches the landmarker instance so both
 * PoseCapturePanel and LivePoseTestPanel share one model.
 */

import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "@mediapipe/tasks-vision";

let cachedLandmarker: PoseLandmarker | null = null;
let loadingPromise: Promise<PoseLandmarker> | null = null;

const WASM_CDN =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const MODEL_CDN =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

/**
 * Get or create the singleton PoseLandmarker.
 * Subsequent calls return the same instance.
 */
export async function getPoseLandmarker(): Promise<PoseLandmarker> {
  if (cachedLandmarker) return cachedLandmarker;
  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(WASM_CDN);
    const landmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_CDN,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });
    cachedLandmarker = landmarker;
    return landmarker;
  })();

  return loadingPromise;
}

/** 33 MediaPipe Pose Landmark names (indexed 0-32). */
export const LANDMARK_NAMES = [
  "Nose",
  "Left Eye Inner",
  "Left Eye",
  "Left Eye Outer",
  "Right Eye Inner",
  "Right Eye",
  "Right Eye Outer",
  "Left Ear",
  "Right Ear",
  "Mouth Left",
  "Mouth Right",
  "Left Shoulder",
  "Right Shoulder",
  "Left Elbow",
  "Right Elbow",
  "Left Wrist",
  "Right Wrist",
  "Left Pinky",
  "Right Pinky",
  "Left Index",
  "Right Index",
  "Left Thumb",
  "Right Thumb",
  "Left Hip",
  "Right Hip",
  "Left Knee",
  "Right Knee",
  "Left Ankle",
  "Right Ankle",
  "Left Heel",
  "Right Heel",
  "Left Foot Index",
  "Right Foot Index",
];

export const N_LANDMARKS = 33;
export const N_FEATURES = N_LANDMARKS * 4; // x, y, z, visibility

/**
 * Flatten a MediaPipe landmarks array into a 132-float feature vector.
 * Order: [lm0_x, lm0_y, lm0_z, lm0_vis, lm1_x, …, lm32_vis]
 */
export function extractLandmarkArray(
  landmarks: { x: number; y: number; z: number; visibility?: number }[],
): number[] {
  const flat: number[] = [];
  for (const lm of landmarks) {
    flat.push(lm.x, lm.y, lm.z, lm.visibility ?? 0);
  }
  return flat;
}

/**
 * Draw landmarks + skeleton connectors on a canvas overlay.
 */
export function drawPoseSkeleton(
  ctx: CanvasRenderingContext2D,
  landmarks: { x: number; y: number; z: number; visibility?: number }[],
  width: number,
  height: number,
) {
  ctx.clearRect(0, 0, width, height);
  const drawingUtils = new DrawingUtils(ctx);

  drawingUtils.drawLandmarks(landmarks, {
    radius: 4,
    color: "#00FF00",
    fillColor: "#00FF00",
  });
  drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
    color: "#00FFFF",
    lineWidth: 2,
  });
}
