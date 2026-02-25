/**
 * AppBuilder — Main 3-panel builder layout.
 * Left: BlockPalette | Center: BlockList (canvas) | Right: BlockEditor
 */

import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router";
import {
  ArrowLeft,
  Eye,
  EyeOff,
  Save,
  Loader2,
  Pencil,
  Palette,
  Globe,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
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

const PANEL_TABS: { key: RightPanel; label: string; Icon: React.ComponentType<{ className?: string }> }[] = [
  { key: "editor", label: "Editor", Icon: Pencil },
  { key: "theme", label: "Theme", Icon: Palette },
  { key: "publish", label: "Publish", Icon: Globe },
];

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
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const initialLoadDone = useRef(false);

  // Fetch block suggestions when app has no blocks
  const hasNoBlocks = app && (!app.blocks || app.blocks.length === 0);
  const { data: suggestions } = useSuggestBlocks(
    hasNoBlocks ? app?.pipelineId : undefined,
  );

  // Load app data into store on initial mount only
  useEffect(() => {
    if (app && !initialLoadDone.current) {
      initialLoadDone.current = true;
      setBlocks(app.blocks ?? []);
      if (app.theme) setTheme(app.theme);
    }
  }, [app]);

  // Reset store on unmount only (separate effect so it doesn't run on every app change)
  useEffect(() => {
    return () => reset();
  }, []);

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
      <div className="h-full flex items-center justify-center bg-gray-50">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="h-8 w-8 animate-spin text-indigo-600" />
          <p className="text-sm text-gray-400">Loading builder...</p>
        </div>
      </div>
    );
  }

  if (!app) {
    return (
      <div className="h-full flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gray-100 flex items-center justify-center">
            <Globe className="h-8 w-8 text-gray-300" />
          </div>
          <h2 className="text-xl font-semibold text-gray-900 mb-2">App Not Found</h2>
          <p className="text-sm text-gray-500 mb-4">This app may have been deleted.</p>
          <button
            onClick={() => navigate("/dashboard")}
            className="text-indigo-600 hover:underline text-sm font-medium"
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
      <div className="h-full flex flex-col bg-gray-50/50">
        <div className="flex items-center justify-between px-5 py-3 bg-white/80 backdrop-blur-xl shadow-[0_1px_3px_rgba(0,0,0,0.04)]">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 bg-amber-50 text-amber-700 px-2.5 py-1 rounded-full text-xs font-medium border border-amber-200/60">
              <Eye className="h-3 w-3" />
              Preview Mode
            </div>
            <span className="text-sm text-gray-400">{app.name}</span>
          </div>
          <button
            onClick={togglePreview}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gray-900 text-white text-sm font-medium hover:bg-gray-800 transition-all shadow-sm"
          >
            <EyeOff className="h-4 w-4" />
            Exit Preview
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-8">
          <div className="max-w-3xl mx-auto space-y-4">
            {blocks.map((block) => (
              <BlockRenderer key={block.id} block={block} mode="preview" theme={theme} />
            ))}
            {blocks.length === 0 && (
              <div className="text-center py-20">
                <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gray-100 flex items-center justify-center">
                  <Eye className="h-8 w-8 text-gray-300" />
                </div>
                <p className="text-gray-400 mb-1">Nothing to preview</p>
                <p className="text-sm text-gray-300">Exit preview to start building.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  const selectedBlock = blocks.find((b) => b.id === selectedBlockId) ?? null;

  return (
    <div className="h-full flex flex-col bg-[#f4f5f7]">
      {/* ── Top Navbar ── */}
      <div className="flex items-center justify-between px-4 py-2 bg-white/80 backdrop-blur-xl shadow-[0_1px_3px_rgba(0,0,0,0.04)] relative z-20">
        <div className="flex items-center gap-2.5">
          <button
            onClick={() => navigate(-1)}
            className="p-2 rounded-xl hover:bg-gray-100/80 transition-all"
          >
            <ArrowLeft className="h-4 w-4 text-gray-500" />
          </button>
          <div className="h-5 w-px bg-gray-200/60" />
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-sm font-semibold text-gray-900">{app.name}</h1>
              {isDirty && (
                <span className="flex items-center gap-1 text-[10px] text-amber-600 bg-amber-50/80 px-1.5 py-0.5 rounded-full border border-amber-200/50 font-medium">
                  Unsaved
                </span>
              )}
            </div>
            <p className="text-[11px] text-gray-400">
              {blocks.length} block{blocks.length !== 1 ? "s" : ""}
              {app.is_published && (
                <span className="text-green-500 ml-1.5">
                  &bull; Published
                </span>
              )}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          {/* Panel tabs */}
          <div className="flex items-center bg-gray-100/70 rounded-xl p-0.5">
            {PANEL_TABS.map(({ key, label, Icon }) => (
              <button
                key={key}
                onClick={() => setRightPanel(key)}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg transition-all duration-200 ${
                  rightPanel === key
                    ? "bg-white text-gray-900 shadow-sm"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                {label}
              </button>
            ))}
          </div>

          <div className="h-5 w-px bg-gray-200/60 mx-0.5" />

          <button
            onClick={togglePreview}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium text-gray-600 hover:bg-gray-100/80 transition-all"
          >
            <Eye className="h-3.5 w-3.5" />
            Preview
          </button>

          <button
            onClick={handleSave}
            disabled={!isDirty || updateApp.isPending}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-medium bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-sm shadow-indigo-200/50 ml-0.5"
          >
            {updateApp.isPending ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Save className="h-3.5 w-3.5" />
            )}
            Save
          </button>
        </div>
      </div>

      {/* ── Three-panel layout ── */}
      <div className="flex-1 flex overflow-hidden p-2 gap-2">
        {/* Left: Block Palette */}
        <div
          className={`bg-white/70 backdrop-blur-sm rounded-2xl shadow-sm overflow-hidden transition-all duration-300 ease-in-out flex flex-col ${
            leftCollapsed ? "w-0 opacity-0 p-0" : "w-56 opacity-100"
          }`}
        >
          <div className="flex-1 overflow-y-auto overflow-x-hidden">
            <BlockPalette />
          </div>
        </div>

        {/* Center: Block List / Canvas */}
        <div className="flex-1 flex flex-col overflow-hidden min-w-0 rounded-2xl bg-white/40 backdrop-blur-sm shadow-sm">
          {/* Canvas toggle bar */}
          <div className="flex items-center justify-between px-3 py-1.5 bg-white/60 backdrop-blur-sm rounded-t-2xl">
            <button
              onClick={() => setLeftCollapsed(!leftCollapsed)}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-medium text-gray-400 hover:text-indigo-600 hover:bg-indigo-50/60 transition-all"
            >
              {leftCollapsed ? (
                <PanelLeftOpen className="h-3.5 w-3.5" />
              ) : (
                <PanelLeftClose className="h-3.5 w-3.5" />
              )}
              {leftCollapsed ? "Blocks" : "Hide"}
            </button>
            <span className="text-[10px] text-gray-300 font-medium">
              {blocks.length} block{blocks.length !== 1 ? "s" : ""} on canvas
            </span>
            <button
              onClick={() => setRightCollapsed(!rightCollapsed)}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-medium text-gray-400 hover:text-indigo-600 hover:bg-indigo-50/60 transition-all"
            >
              {rightCollapsed ? "Editor" : "Hide"}
              {rightCollapsed ? (
                <PanelRightOpen className="h-3.5 w-3.5" />
              ) : (
                <PanelRightClose className="h-3.5 w-3.5" />
              )}
            </button>
          </div>
          <div
            className="flex-1 overflow-y-auto p-6"
            style={{
              background: "linear-gradient(160deg, #f8f9fb 0%, #f1f3f8 40%, #eef0fa 100%)",
              backgroundImage: `linear-gradient(160deg, #f8f9fb 0%, #f1f3f8 40%, #eef0fa 100%), radial-gradient(circle, #e2e5ee 0.8px, transparent 0.8px)`,
              backgroundSize: "100% 100%, 24px 24px",
            }}
          >
            <BlockList blocks={blocks} selectedBlockId={selectedBlockId} theme={theme} />
          </div>
        </div>

        {/* Right: Editor / Theme / Publish */}
        <div
          className={`bg-white/70 backdrop-blur-sm rounded-2xl shadow-sm overflow-hidden transition-all duration-300 ease-in-out flex flex-col ${
            rightCollapsed ? "w-0 opacity-0 p-0" : "w-80 opacity-100"
          }`}
        >
          <div className="flex-1 overflow-y-auto overflow-x-hidden">
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
    </div>
  );
}
