/**
 * Text Block — Paragraph or description text.
 */

import type { BlockRenderProps } from "../BlockRenderer";
import type { TextConfig } from "../../types/appBuilder";

export default function TextBlock({ block }: BlockRenderProps) {
  const config = block.config as TextConfig;

  const sizeClass =
    config.size === "sm"
      ? "text-sm"
      : config.size === "lg"
        ? "text-lg"
        : "text-base";

  const alignClass =
    config.alignment === "center"
      ? "text-center"
      : config.alignment === "right"
        ? "text-right"
        : "text-left";

  return (
    <div className={`${sizeClass} ${alignClass} text-gray-600 leading-relaxed`}>
      {config.content}
    </div>
  );
}
