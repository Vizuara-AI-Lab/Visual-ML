/**
 * Divider Block — Visual separator.
 */

import type { BlockRenderProps } from "../BlockRenderer";
import type { DividerConfig } from "../../types/appBuilder";

export default function DividerBlock({ block }: BlockRenderProps) {
  const config = block.config as DividerConfig;

  if (config.style === "space") {
    return <div className="h-8" />;
  }

  if (config.style === "dots") {
    return (
      <div className="flex items-center justify-center gap-2 py-4">
        <span className="h-1.5 w-1.5 rounded-full bg-gray-300" />
        <span className="h-1.5 w-1.5 rounded-full bg-gray-300" />
        <span className="h-1.5 w-1.5 rounded-full bg-gray-300" />
      </div>
    );
  }

  return <hr className="border-gray-200 my-2" />;
}
