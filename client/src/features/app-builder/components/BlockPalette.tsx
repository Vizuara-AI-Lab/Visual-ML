/**
 * BlockPalette — Left panel showing available block types grouped by collapsible categories.
 * Click to add a block to the canvas.
 */

import { useState } from "react";
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
  Plus,
  MoveVertical,
  AlertTriangle,
  Code,
  Video,
  ChevronDown,
  Layers,
  LogIn,
  MonitorPlay,
  Film,
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
  MoveVertical,
  AlertTriangle,
  Code,
  Video,
};

// Category icons and accent colors
const CATEGORY_META: Record<
  string,
  {
    Icon: React.ComponentType<{ className?: string }>;
    accentBg: string;
    accentText: string;
    hoverBg: string;
    iconBg: string;
    iconHover: string;
    iconText: string;
    iconTextHover: string;
  }
> = {
  Layout: {
    Icon: Layers,
    accentBg: "bg-indigo-50",
    accentText: "text-indigo-600",
    hoverBg: "hover:bg-indigo-50/60",
    iconBg: "bg-indigo-50",
    iconHover: "group-hover:bg-indigo-100",
    iconText: "text-indigo-400",
    iconTextHover: "group-hover:text-indigo-600",
  },
  Input: {
    Icon: LogIn,
    accentBg: "bg-emerald-50",
    accentText: "text-emerald-600",
    hoverBg: "hover:bg-emerald-50/60",
    iconBg: "bg-emerald-50",
    iconHover: "group-hover:bg-emerald-100",
    iconText: "text-emerald-400",
    iconTextHover: "group-hover:text-emerald-600",
  },
  Output: {
    Icon: MonitorPlay,
    accentBg: "bg-amber-50",
    accentText: "text-amber-600",
    hoverBg: "hover:bg-amber-50/60",
    iconBg: "bg-amber-50",
    iconHover: "group-hover:bg-amber-100",
    iconText: "text-amber-400",
    iconTextHover: "group-hover:text-amber-600",
  },
  "Media & Content": {
    Icon: Film,
    accentBg: "bg-violet-50",
    accentText: "text-violet-600",
    hoverBg: "hover:bg-violet-50/60",
    iconBg: "bg-violet-50",
    iconHover: "group-hover:bg-violet-100",
    iconText: "text-violet-400",
    iconTextHover: "group-hover:text-violet-600",
  },
};

const CATEGORIES: { label: string; types: BlockType[] }[] = [
  {
    label: "Layout",
    types: ["hero", "text", "divider", "spacer", "image"],
  },
  {
    label: "Input",
    types: ["file_upload", "input_fields", "submit_button"],
  },
  {
    label: "Output",
    types: ["results_display", "metrics_card"],
  },
  {
    label: "Media & Content",
    types: ["alert", "code", "video_embed"],
  },
];

export default function BlockPalette() {
  const addBlock = useAppBuilderStore((s) => s.addBlock);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const toggle = (label: string) =>
    setCollapsed((prev) => ({ ...prev, [label]: !prev[label] }));

  return (
    <div className="p-3 space-y-2">
      {/* Header */}
      <div className="px-1 pt-1 pb-1.5">
        <h2 className="text-xs font-bold text-gray-700 tracking-wide">
          Blocks
        </h2>
        <p className="text-[10px] text-gray-400 mt-0.5">
          Click to add to canvas
        </p>
      </div>

      {CATEGORIES.map((category) => {
        const meta = CATEGORY_META[category.label];
        const isCollapsed = collapsed[category.label];
        const CatIcon = meta?.Icon;

        return (
          <div
            key={category.label}
            className="rounded-xl overflow-hidden"
          >
            {/* Category header — clickable to collapse */}
            <button
              onClick={() => toggle(category.label)}
              className="w-full flex items-center gap-2 px-2.5 py-2 text-left transition-all rounded-xl hover:bg-gray-50/80"
            >
              {CatIcon && (
                <div className={`w-5 h-5 rounded-lg flex items-center justify-center ${meta.accentBg}`}>
                  <CatIcon className={`h-3 w-3 ${meta.accentText}`} />
                </div>
              )}
              <span className="text-[11px] font-semibold text-gray-600 flex-1">
                {category.label}
              </span>
              <span className="text-[9px] text-gray-300 mr-1">{category.types.length}</span>
              <ChevronDown
                className={`h-3 w-3 text-gray-300 transition-transform duration-200 ${
                  isCollapsed ? "-rotate-90" : ""
                }`}
              />
            </button>

            {/* Block items */}
            {!isCollapsed && (
              <div className="px-1 pb-1.5 space-y-0.5">
                {category.types.map((type) => {
                  const def = BLOCK_DEFINITIONS.find((d) => d.type === type);
                  if (!def) return null;
                  const Icon = ICON_MAP[def.icon];
                  return (
                    <button
                      key={def.type}
                      onClick={() => addBlock(def.type as BlockType)}
                      className={`w-full flex items-center gap-2.5 px-2 py-1.5 rounded-lg text-left ${meta.hoverBg} transition-all group`}
                    >
                      <div
                        className={`w-7 h-7 rounded-lg ${meta.iconBg} ${meta.iconHover} flex items-center justify-center shrink-0 transition-colors`}
                      >
                        {Icon && (
                          <Icon
                            className={`h-3.5 w-3.5 ${meta.iconText} ${meta.iconTextHover} transition-colors`}
                          />
                        )}
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="text-xs font-medium text-gray-700 truncate leading-tight">
                          {def.label}
                        </p>
                        <p className="text-[10px] text-gray-400 truncate leading-tight">
                          {def.description}
                        </p>
                      </div>
                      <Plus className="h-3 w-3 text-gray-300 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
