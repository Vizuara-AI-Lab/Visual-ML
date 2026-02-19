/**
 * Alert Block — Info, warning, success, or error notice.
 */

import { Info, AlertTriangle, CheckCircle, XCircle } from "lucide-react";
import type { BlockRenderProps } from "../BlockRenderer";
import type { AlertConfig } from "../../types/appBuilder";

const VARIANT_STYLES = {
  info: {
    bg: "bg-blue-50",
    border: "border-blue-200",
    titleColor: "text-blue-800",
    textColor: "text-blue-700",
    Icon: Info,
    iconColor: "text-blue-500",
  },
  warning: {
    bg: "bg-amber-50",
    border: "border-amber-200",
    titleColor: "text-amber-800",
    textColor: "text-amber-700",
    Icon: AlertTriangle,
    iconColor: "text-amber-500",
  },
  success: {
    bg: "bg-green-50",
    border: "border-green-200",
    titleColor: "text-green-800",
    textColor: "text-green-700",
    Icon: CheckCircle,
    iconColor: "text-green-500",
  },
  error: {
    bg: "bg-red-50",
    border: "border-red-200",
    titleColor: "text-red-800",
    textColor: "text-red-700",
    Icon: XCircle,
    iconColor: "text-red-500",
  },
};

export default function AlertBlock({ block }: BlockRenderProps) {
  const config = block.config as AlertConfig;
  const style = VARIANT_STYLES[config.variant];

  return (
    <div className={`${style.bg} ${style.border} border rounded-lg p-4`}>
      <div className="flex gap-3">
        {config.showIcon && (
          <style.Icon className={`h-5 w-5 ${style.iconColor} shrink-0 mt-0.5`} />
        )}
        <div>
          {config.title && (
            <p className={`text-sm font-semibold ${style.titleColor} mb-1`}>
              {config.title}
            </p>
          )}
          <p className={`text-sm ${style.textColor}`}>{config.message}</p>
        </div>
      </div>
    </div>
  );
}
