/**
 * Hero Block — Page header with title, subtitle, and optional gradient.
 */

import type { BlockRenderProps } from "../BlockRenderer";
import type { HeroConfig } from "../../types/appBuilder";

export default function HeroBlock({ block, theme }: BlockRenderProps) {
  const config = block.config as HeroConfig;
  const align =
    config.alignment === "center"
      ? "text-center"
      : config.alignment === "right"
        ? "text-right"
        : "text-left";

  return (
    <div
      className={`rounded-xl px-8 py-12 ${align} ${
        config.showGradient
          ? "bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-500 text-white"
          : "bg-white border"
      }`}
      style={
        config.showGradient
          ? { background: `linear-gradient(135deg, ${theme.primaryColor}, #ec4899)` }
          : undefined
      }
    >
      <h1
        className={`text-3xl font-bold mb-3 ${
          config.showGradient ? "text-white" : "text-gray-900"
        }`}
      >
        {config.title}
      </h1>
      {config.subtitle && (
        <p
          className={`text-lg ${
            config.showGradient ? "text-white/80" : "text-gray-500"
          }`}
        >
          {config.subtitle}
        </p>
      )}
    </div>
  );
}
