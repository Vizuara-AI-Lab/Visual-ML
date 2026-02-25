/**
 * ImageUploadTest — upload a single image and test it against a trained model.
 * Converts to grayscale pixel array at model's target resolution for inference.
 */

import { useRef, useState, useCallback } from "react";
import { Upload, Image as ImageIcon, Loader2, X } from "lucide-react";

interface Prediction {
  class_name: string;
  confidence: number;
  all_scores: { class_name: string; score: number }[];
}

interface ImageUploadTestProps {
  imageWidth: number;
  imageHeight: number;
  classNames: string[];
  onPredict: (pixels: number[]) => Promise<Prediction>;
}

export const ImageUploadTest = ({
  imageWidth,
  imageHeight,
  classNames,
  onPredict,
}: ImageUploadTestProps) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [pixels, setPixels] = useState<number[] | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [fileName, setFileName] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const processFile = useCallback(
    (file: File) => {
      setFileName(file.name);
      setPrediction(null);

      const img = new window.Image();
      img.onload = () => {
        // Convert to target size grayscale
        const canvas = document.createElement("canvas");
        canvas.width = imageWidth;
        canvas.height = imageHeight;
        const ctx = canvas.getContext("2d")!;
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, imageWidth, imageHeight);

        // Center-crop to square
        const side = Math.min(img.width, img.height);
        const sx = (img.width - side) / 2;
        const sy = (img.height - side) / 2;
        ctx.drawImage(img, sx, sy, side, side, 0, 0, imageWidth, imageHeight);

        const data = ctx.getImageData(0, 0, imageWidth, imageHeight).data;
        const pxArr: number[] = [];
        for (let i = 0; i < data.length; i += 4) {
          pxArr.push(
            Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]),
          );
        }
        setPixels(pxArr);

        // High-res preview
        const prevCanvas = document.createElement("canvas");
        const PREVIEW = 224;
        prevCanvas.width = PREVIEW;
        prevCanvas.height = PREVIEW;
        const pctx = prevCanvas.getContext("2d")!;
        pctx.imageSmoothingEnabled = true;
        pctx.imageSmoothingQuality = "high";
        pctx.drawImage(img, sx, sy, side, side, 0, 0, PREVIEW, PREVIEW);
        setPreview(prevCanvas.toDataURL("image/png"));

        URL.revokeObjectURL(img.src);
      };
      img.src = URL.createObjectURL(file);
    },
    [imageWidth, imageHeight],
  );

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const file = files[0];
      if (!file.type.startsWith("image/")) return;
      processFile(file);
    },
    [processFile],
  );

  const handlePredict = useCallback(async () => {
    if (!pixels) return;
    setPredicting(true);
    try {
      const result = await onPredict(pixels);
      setPrediction(result);
    } catch (err) {
      console.error("Prediction failed:", err);
    } finally {
      setPredicting(false);
    }
  }, [pixels, onPredict]);

  const clearImage = useCallback(() => {
    setPreview(null);
    setPixels(null);
    setPrediction(null);
    setFileName("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  const topScore = prediction
    ? Math.max(...prediction.all_scores.map((s) => s.score))
    : 0;

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row gap-4">
        {/* Upload area */}
        <div className="flex flex-col items-center gap-3 min-w-[260px]">
          {!preview ? (
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                handleFiles(e.dataTransfer.files);
              }}
              onClick={() => fileInputRef.current?.click()}
              className={`w-full aspect-square max-w-[280px] rounded-xl border-2 border-dashed flex flex-col items-center justify-center gap-3 cursor-pointer transition-colors ${
                dragOver
                  ? "border-violet-400 bg-violet-50"
                  : "border-slate-300 bg-slate-50 hover:border-violet-300 hover:bg-violet-50/30"
              }`}
            >
              <Upload className={`w-10 h-10 ${dragOver ? "text-violet-500" : "text-slate-400"}`} />
              <div className="text-center">
                <p className="text-sm font-medium text-slate-600">
                  Drop an image here
                </p>
                <p className="text-xs text-slate-400 mt-0.5">
                  or click to browse (PNG, JPG)
                </p>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => handleFiles(e.target.files)}
              />
            </div>
          ) : (
            <div className="relative">
              <div className="rounded-xl border-2 border-slate-300 overflow-hidden shadow-lg">
                <img
                  src={preview}
                  alt="Uploaded"
                  className="w-[280px] h-[280px] object-cover"
                />
              </div>
              <button
                onClick={clearImage}
                className="absolute -top-2 -right-2 p-1 bg-red-500 hover:bg-red-600 text-white rounded-full shadow transition-colors"
              >
                <X className="w-3.5 h-3.5" />
              </button>
              <p className="text-xs text-slate-500 text-center mt-2 truncate max-w-[280px]">
                {fileName}
              </p>
            </div>
          )}

          {preview && (
            <button
              onClick={handlePredict}
              disabled={!pixels || predicting}
              className="w-full max-w-[280px] flex items-center justify-center gap-2 py-2.5 bg-violet-600 hover:bg-violet-700 disabled:bg-slate-200 disabled:text-slate-400 text-white rounded-lg text-sm font-semibold transition-colors"
            >
              {predicting ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <ImageIcon className="w-4 h-4" />
              )}
              {predicting ? "Predicting…" : "Predict"}
            </button>
          )}

          <p className="text-[10px] text-slate-400 text-center">
            Image will be converted to {imageWidth}×{imageHeight}px grayscale
          </p>
        </div>

        {/* Prediction result */}
        <div className="flex-1 min-w-[200px]">
          {prediction ? (
            <div className="space-y-3">
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
              <ImageIcon className="w-8 h-8 text-slate-300 mx-auto mb-2" />
              <p className="text-sm font-medium text-slate-500">Upload an image to test</p>
              <p className="text-xs text-slate-400 mt-1">
                {classNames.length > 0
                  ? `Classes: ${classNames.join(", ")}`
                  : "Upload an image and click Predict"}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
