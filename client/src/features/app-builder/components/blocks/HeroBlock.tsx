/**
 * Hero Block — Page header with title, subtitle, gradient, and optional CTA button.
 * Uses per-block gradient colors (not global theme) so multiple hero blocks are independent.
 */

import type { BlockRenderProps } from "../BlockRenderer";
import type { HeroConfig } from "../../types/appBuilder";

const SIZE_CLASSES: Record<string, { title: string; subtitle: string; padding: string }> = {
  sm: { title: "text-xl", subtitle: "text-sm", padding: "px-6 py-8" },
  md: { title: "text-2xl", subtitle: "text-base", padding: "px-8 py-10" },
  lg: { title: "text-3xl", subtitle: "text-lg", padding: "px-8 py-12" },
  xl: { title: "text-4xl", subtitle: "text-xl", padding: "px-10 py-16" },
};

export default function HeroBlock({ block }: BlockRenderProps) {
  const config = block.config as HeroConfig;
  const align =
    config.alignment === "center"
      ? "text-center"
      : config.alignment === "right"
        ? "text-right"
        : "text-left";

  const sizes = SIZE_CLASSES[config.size] || SIZE_CLASSES.lg;
  const from = config.gradientFrom || "#6366f1";
  const to = config.gradientTo || "#ec4899";

  const alignItems =
    config.alignment === "center"
      ? "items-center"
      : config.alignment === "right"
        ? "items-end"
        : "items-start";

  return (
    <div
      className={`rounded-xl ${sizes.padding} ${align} relative overflow-hidden`}
      style={
        config.showGradient
          ? { background: `linear-gradient(135deg, ${from}, ${to})` }
          : { backgroundColor: "#ffffff", border: "1px solid #e5e7eb" }
      }
    >
      {config.showGradient && (
        <div
          className="absolute inset-0 opacity-20"
          style={{
            backgroundImage:
              "radial-gradient(circle at 20% 50%, rgba(255,255,255,0.3) 0%, transparent 50%), radial-gradient(circle at 80% 20%, rgba(255,255,255,0.2) 0%, transparent 40%)",
          }}
        />
      )}

      <div className={`relative z-10 flex flex-col ${alignItems}`}>
        <h1
          className={`${sizes.title} font-bold mb-3 leading-tight ${
            config.showGradient ? "text-white" : "text-gray-900"
          }`}
        >
          {config.title}
        </h1>
        {config.subtitle && (
          <p
            className={`${sizes.subtitle} max-w-xl leading-relaxed ${
              config.showGradient ? "text-white/80" : "text-gray-500"
            }`}
          >
            {config.subtitle}
          </p>
        )}
        {config.buttonText && (
          <button
            className={`mt-5 px-6 py-2.5 rounded-lg text-sm font-semibold transition-all shadow-md hover:shadow-lg ${
              config.showGradient
                ? "bg-white/20 backdrop-blur-sm text-white border border-white/30 hover:bg-white/30"
                : "bg-gray-900 text-white hover:bg-gray-800"
            }`}
          >
            {config.buttonText}
          </button>
        )}
      </div>
    </div>
  );
}
