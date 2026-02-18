/**
 * Submit Button Block — Triggers pipeline execution.
 */

import { Loader2, Play } from "lucide-react";
import type { BlockRenderProps } from "../BlockRenderer";
import type { SubmitButtonConfig } from "../../types/appBuilder";

export default function SubmitButtonBlock({
  block,
  mode,
  theme,
  isExecuting,
  onSubmit,
}: BlockRenderProps) {
  const config = block.config as SubmitButtonConfig;
  const isInteractive = mode === "live";

  const baseClasses =
    "w-full py-3 px-6 rounded-lg font-medium text-sm transition-all flex items-center justify-center gap-2";

  const variantClasses =
    config.variant === "gradient"
      ? "text-white bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 shadow-lg"
      : config.variant === "secondary"
        ? "text-gray-700 bg-gray-100 hover:bg-gray-200 border"
        : "text-white hover:opacity-90";

  const style =
    config.variant === "primary"
      ? { backgroundColor: theme.primaryColor }
      : undefined;

  return (
    <button
      onClick={isInteractive ? onSubmit : undefined}
      disabled={!isInteractive || isExecuting}
      className={`${baseClasses} ${variantClasses} disabled:opacity-50 disabled:cursor-not-allowed`}
      style={style}
    >
      {isExecuting ? (
        <>
          <Loader2 className="h-4 w-4 animate-spin" />
          {config.loadingText}
        </>
      ) : (
        <>
          <Play className="h-4 w-4" />
          {config.label}
        </>
      )}
    </button>
  );
}
