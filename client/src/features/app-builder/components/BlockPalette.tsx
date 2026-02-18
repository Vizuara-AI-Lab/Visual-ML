/**
 * BlockPalette — Left panel showing available block types.
 * Click to add a block to the canvas.
 */

import {
  Layout,
  Type,
  Upload,
  FormInput,
  Play,
  Table,
  BarChart3,
  Minus,
  Image,
} from "lucide-react";
import { BLOCK_DEFINITIONS } from "../config/blockDefinitions";
import { useAppBuilderStore } from "../store/appBuilderStore";
import type { BlockType } from "../types/appBuilder";

const ICON_MAP: Record<string, React.ComponentType<{ className?: string }>> = {
  Layout,
  Type,
  Upload,
  FormInput,
  Play,
  Table,
  BarChart3,
  Minus,
  Image,
};

export default function BlockPalette() {
  const addBlock = useAppBuilderStore((s) => s.addBlock);

  return (
    <div className="p-4">
      <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
        Blocks
      </h2>
      <div className="space-y-1.5">
        {BLOCK_DEFINITIONS.map((def) => {
          const Icon = ICON_MAP[def.icon];
          return (
            <button
              key={def.type}
              onClick={() => addBlock(def.type as BlockType)}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left hover:bg-indigo-50 hover:text-indigo-700 transition-colors group"
            >
              {Icon && (
                <Icon className="h-4 w-4 text-gray-400 group-hover:text-indigo-500 shrink-0" />
              )}
              <div className="min-w-0">
                <p className="text-sm font-medium text-gray-700 group-hover:text-indigo-700 truncate">
                  {def.label}
                </p>
                <p className="text-xs text-gray-400 truncate">{def.description}</p>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
