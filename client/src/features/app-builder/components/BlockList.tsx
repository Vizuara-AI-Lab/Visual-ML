/**
 * BlockList — Center panel showing the ordered list of blocks.
 * Supports selection, reorder (up/down), and delete.
 */

import {
  ChevronUp,
  ChevronDown,
  Trash2,
  Link,
  Layers,
} from "lucide-react";
import { useAppBuilderStore } from "../store/appBuilderStore";
import BlockRenderer from "./BlockRenderer";
import type { AppBlock, AppTheme } from "../types/appBuilder";

interface BlockListProps {
  blocks: AppBlock[];
  selectedBlockId: string | null;
  theme: AppTheme;
}

export default function BlockList({ blocks, selectedBlockId, theme }: BlockListProps) {
  const { selectBlock, moveBlock, removeBlock } = useAppBuilderStore();

  if (blocks.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center max-w-xs">
          <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-white border-2 border-dashed border-gray-200 flex items-center justify-center">
            <Layers className="h-7 w-7 text-gray-300" />
          </div>
          <p className="text-sm font-medium text-gray-500 mb-1">No blocks yet</p>
          <p className="text-xs text-gray-400 leading-relaxed">
            Click a block in the left palette to start building your app.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto space-y-3">
      {blocks.map((block, index) => {
        const isSelected = block.id === selectedBlockId;
        return (
          <div
            key={block.id}
            onClick={() => selectBlock(block.id)}
            className={`relative group rounded-xl transition-all cursor-pointer ${
              isSelected
                ? "ring-2 ring-indigo-500 ring-offset-2"
                : "hover:ring-2 hover:ring-gray-300 hover:ring-offset-1"
            }`}
          >
            {/* Block controls — left side */}
            <div className="absolute -left-10 top-1/2 -translate-y-1/2 flex flex-col gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  moveBlock(block.id, "up");
                }}
                disabled={index === 0}
                className="p-1 rounded-md hover:bg-white hover:shadow-sm disabled:opacity-30 transition-all"
              >
                <ChevronUp className="h-3.5 w-3.5 text-gray-400" />
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  moveBlock(block.id, "down");
                }}
                disabled={index === blocks.length - 1}
                className="p-1 rounded-md hover:bg-white hover:shadow-sm disabled:opacity-30 transition-all"
              >
                <ChevronDown className="h-3.5 w-3.5 text-gray-400" />
              </button>
            </div>

            {/* Delete button */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                removeBlock(block.id);
              }}
              className="absolute -right-3 -top-3 p-1.5 rounded-full bg-white shadow-md opacity-0 group-hover:opacity-100 transition-all hover:bg-red-50 hover:shadow-lg z-20"
            >
              <Trash2 className="h-3.5 w-3.5 text-red-500" />
            </button>

            {/* Node mapping badge — always visible */}
            {block.nodeLabel && (
              <div className="absolute -right-3 top-8 z-10">
                <div className="flex items-center gap-1 bg-indigo-50 text-indigo-700 text-[10px] font-medium px-2 py-0.5 rounded-full border border-indigo-200 whitespace-nowrap shadow-sm">
                  <Link className="h-2.5 w-2.5" />
                  {block.nodeLabel}
                </div>
              </div>
            )}

            {/* Block type label — visible when selected */}
            {isSelected && (
              <div className="absolute -top-2.5 left-3 z-10">
                <span className="text-[10px] font-medium bg-indigo-600 text-white px-2 py-0.5 rounded-full shadow-sm capitalize">
                  {block.type.replace(/_/g, " ")}
                </span>
              </div>
            )}

            {/* Block render */}
            <BlockRenderer block={block} mode="edit" theme={theme} />
          </div>
        );
      })}
    </div>
  );
}
