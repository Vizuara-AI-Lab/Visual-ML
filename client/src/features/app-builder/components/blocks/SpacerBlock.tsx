/**
 * Spacer Block — Configurable vertical space.
 */

import { MoveVertical } from "lucide-react";
import type { BlockRenderProps } from "../BlockRenderer";
import type { SpacerConfig } from "../../types/appBuilder";

export default function SpacerBlock({ block, mode }: BlockRenderProps) {
  const config = block.config as SpacerConfig;

  if (mode === "edit") {
    return (
      <div
        className="flex items-center justify-center border border-dashed border-gray-200 rounded-lg bg-gray-50/50"
        style={{ height: `${config.height}px` }}
      >
        <div className="flex items-center gap-1.5 text-gray-300">
          <MoveVertical className="h-3.5 w-3.5" />
          <span className="text-xs">{config.height}px</span>
        </div>
      </div>
    );
  }

  return <div style={{ height: `${config.height}px` }} />;
}
