/**
 * Image Block — Static image display.
 */

import { ImageIcon } from "lucide-react";
import type { BlockRenderProps } from "../BlockRenderer";
import type { ImageConfig } from "../../types/appBuilder";

export default function ImageBlock({ block }: BlockRenderProps) {
  const config = block.config as ImageConfig;

  const widthClass =
    config.width === "sm"
      ? "max-w-xs"
      : config.width === "md"
        ? "max-w-md"
        : config.width === "lg"
          ? "max-w-lg"
          : "w-full";

  const imageSrc = config.base64Data || config.url;

  if (!imageSrc) {
    return (
      <div
        className={`${widthClass} mx-auto bg-gray-100 rounded-lg flex items-center justify-center py-12`}
      >
        <div className="text-center text-gray-400">
          <ImageIcon className="h-8 w-8 mx-auto mb-2" />
          <p className="text-sm">No image set</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`${widthClass} mx-auto`}>
      <img
        src={imageSrc}
        alt={config.alt}
        className="w-full rounded-lg"
      />
    </div>
  );
}
