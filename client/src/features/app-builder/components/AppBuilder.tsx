/**
 * AppBuilder — Main 3-panel builder layout.
 * Left: BlockPalette | Center: BlockList (canvas) | Right: BlockEditor
 */

import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import {
  ArrowLeft,
  Eye,
  EyeOff,
  Save,
  Loader2,
} from "lucide-react";
import { toast } from "react-hot-toast";
import { useCustomApp, useUpdateApp, useSuggestBlocks } from "../hooks/useAppBuilder";
import { useAppBuilderStore } from "../store/appBuilderStore";
import BlockPalette from "./BlockPalette";
import BlockList from "./BlockList";
import BlockEditor from "./BlockEditor";
import ThemePanel from "./ThemePanel";
import PublishPanel from "./PublishPanel";
import BlockRenderer from "./BlockRenderer";
import type { AppBlock } from "../types/appBuilder";

interface AppBuilderProps {
  appId: number | undefined;
}

type RightPanel = "editor" | "theme" | "publish";

export default function AppBuilder({ appId }: AppBuilderProps) {
  const navigate = useNavigate();
  const { data: app, isLoading } = useCustomApp(appId);
  const updateApp = useUpdateApp(appId ?? 0);

  const {
    blocks,
    selectedBlockId,
    theme,
    isDirty,
    isPreviewMode,
    setBlocks,
    setBlocksFromSuggestion,
    setTheme,
    togglePreview,
    markClean,
    reset,
  } = useAppBuilderStore();

  const [rightPanel, setRightPanel] = useState<RightPanel>("editor");

  // Fetch block suggestions when app has no blocks
  const hasNoBlocks = app && (!app.blocks || app.blocks.length === 0);
  const { data: suggestions } = useSuggestBlocks(
    hasNoBlocks ? app?.pipelineId : undefined,
  );

  // Load app data into store on mount
  useEffect(() => {
    if (app) {
      setBlocks(app.blocks ?? []);
      if (app.theme) setTheme(app.theme);
    }
    return () => reset();
  }, [app]);

  // Auto-populate builder from pipeline suggestions when app is empty
  useEffect(() => {
    if (suggestions && hasNoBlocks && blocks.length === 0) {
      setBlocksFromSuggestion(suggestions.blocks);
      toast.success(`Auto-generated ${suggestions.blocks.length} blocks from pipeline`);
    }
  }, [suggestions, hasNoBlocks]);

  const handleSave = async () => {
    if (!appId) return;
    try {
      await updateApp.mutateAsync({ blocks, theme });
      markClean();
      toast.success("App saved");
    } catch {
      toast.error("Failed to save");
    }
  };

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-indigo-600" />
      </div>
    );
  }

  if (!app) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-gray-900 mb-2">App Not Found</h2>
          <button
            onClick={() => navigate("/dashboard")}
            className="text-indigo-600 hover:underline"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  // Preview mode — full-width render
  if (isPreviewMode) {
    return (
      <div className="h-full flex flex-col">
        <div className="flex items-center justify-between px-4 py-3 border-b bg-white">
          <h2 className="text-sm font-medium text-gray-600">Preview Mode</h2>
          <button
            onClick={togglePreview}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-indigo-600 text-white text-sm hover:bg-indigo-700 transition-colors"
          >
            <EyeOff className="h-4 w-4" />
            Exit Preview
          </button>
        </div>
        <div className="flex-1 overflow-y-auto bg-gray-50 p-8">
          <div className="max-w-3xl mx-auto space-y-4">
            {blocks.map((block) => (
              <BlockRenderer key={block.id} block={block} mode="preview" theme={theme} />
            ))}
            {blocks.length === 0 && (
              <p className="text-center text-gray-400 py-20">
                No blocks added yet. Exit preview to start building.
              </p>
            )}
          </div>
        </div>
      </div>
    );
  }

  const selectedBlock = blocks.find((b) => b.id === selectedBlockId) ?? null;

  return (
    <div className="h-full flex flex-col">
      {/* Top toolbar */}
      <div className="flex items-center justify-between px-4 py-3 border-b bg-white">
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate(-1)}
            className="p-1.5 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <ArrowLeft className="h-5 w-5 text-gray-600" />
          </button>
          <div>
            <h1 className="text-lg font-semibold text-gray-900">{app.name}</h1>
            <p className="text-xs text-gray-500">App Builder</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Tab switches for right panel */}
          {(["editor", "theme", "publish"] as RightPanel[]).map((tab) => (
            <button
              key={tab}
              onClick={() => setRightPanel(tab)}
              className={`px-3 py-1.5 text-sm rounded-lg transition-colors capitalize ${
                rightPanel === tab
                  ? "bg-indigo-100 text-indigo-700"
                  : "text-gray-600 hover:bg-gray-100"
              }`}
            >
              {tab}
            </button>
          ))}

          <div className="w-px h-6 bg-gray-200 mx-1" />

          <button
            onClick={togglePreview}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm text-gray-600 hover:bg-gray-100 transition-colors"
          >
            <Eye className="h-4 w-4" />
            Preview
          </button>

          <button
            onClick={handleSave}
            disabled={!isDirty || updateApp.isPending}
            className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-sm bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {updateApp.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Save className="h-4 w-4" />
            )}
            Save
          </button>
        </div>
      </div>

      {/* Three-panel layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Block Palette */}
        <div className="w-56 border-r bg-white overflow-y-auto">
          <BlockPalette />
        </div>

        {/* Center: Block List / Canvas */}
        <div className="flex-1 overflow-y-auto bg-gray-50 p-6">
          <BlockList blocks={blocks} selectedBlockId={selectedBlockId} theme={theme} />
        </div>

        {/* Right: Editor / Theme / Publish */}
        <div className="w-80 border-l bg-white overflow-y-auto">
          {rightPanel === "editor" && (
            <BlockEditor block={selectedBlock} />
          )}
          {rightPanel === "theme" && <ThemePanel />}
          {rightPanel === "publish" && (
            <PublishPanel app={app} />
          )}
        </div>
      </div>
    </div>
  );
}
