/**
 * Custom App Builder — Zustand Store
 * Manages builder UI state: blocks, selection, theme, dirty flag.
 */

import { create } from "zustand";
import type { AppBlock, AppTheme, BlockType, BlockStyleConfig } from "../types/appBuilder";
import { getDefaultConfig } from "../config/blockDefinitions";

interface AppBuilderState {
  // Builder state
  blocks: AppBlock[];
  selectedBlockId: string | null;
  theme: AppTheme;
  isDirty: boolean;
  isPreviewMode: boolean;

  // Actions — blocks
  setBlocks: (blocks: AppBlock[]) => void;
  addBlock: (type: BlockType) => void;
  updateBlock: (id: string, config: Partial<AppBlock["config"]>) => void;
  updateBlockStyle: (id: string, style: Partial<BlockStyleConfig>) => void;
  removeBlock: (id: string) => void;
  moveBlock: (id: string, direction: "up" | "down") => void;
  selectBlock: (id: string | null) => void;

  // Actions — theme
  setTheme: (theme: Partial<AppTheme>) => void;

  // Actions — suggestions
  setBlocksFromSuggestion: (blocks: AppBlock[]) => void;

  // Actions — UI
  togglePreview: () => void;
  markClean: () => void;
  reset: () => void;
}

const DEFAULT_THEME: AppTheme = {
  primaryColor: "#6366f1",
  fontFamily: "Inter",
  darkMode: false,
};

let blockCounter = 0;
function generateBlockId(): string {
  blockCounter += 1;
  return `block_${Date.now()}_${blockCounter}`;
}

export const useAppBuilderStore = create<AppBuilderState>()((set, get) => ({
  blocks: [],
  selectedBlockId: null,
  theme: { ...DEFAULT_THEME },
  isDirty: false,
  isPreviewMode: false,

  setBlocks: (blocks) => set({ blocks, isDirty: false }),

  addBlock: (type) => {
    const { blocks } = get();
    const newBlock: AppBlock = {
      id: generateBlockId(),
      type,
      config: getDefaultConfig(type),
      order: blocks.length,
    };
    set({
      blocks: [...blocks, newBlock],
      selectedBlockId: newBlock.id,
      isDirty: true,
    });
  },

  updateBlock: (id, config) => {
    const { blocks } = get();
    set({
      blocks: blocks.map((b) =>
        b.id === id ? { ...b, config: { ...b.config, ...config } } : b
      ),
      isDirty: true,
    });
  },

  updateBlockStyle: (id, style) => {
    const { blocks } = get();
    set({
      blocks: blocks.map((b) =>
        b.id === id
          ? { ...b, style: { ...(b.style || {}), ...style } }
          : b
      ),
      isDirty: true,
    });
  },

  removeBlock: (id) => {
    const { blocks, selectedBlockId } = get();
    const filtered = blocks
      .filter((b) => b.id !== id)
      .map((b, i) => ({ ...b, order: i }));
    set({
      blocks: filtered,
      selectedBlockId: selectedBlockId === id ? null : selectedBlockId,
      isDirty: true,
    });
  },

  moveBlock: (id, direction) => {
    const { blocks } = get();
    const idx = blocks.findIndex((b) => b.id === id);
    if (idx < 0) return;
    const newIdx = direction === "up" ? idx - 1 : idx + 1;
    if (newIdx < 0 || newIdx >= blocks.length) return;

    const updated = [...blocks];
    [updated[idx], updated[newIdx]] = [updated[newIdx], updated[idx]];
    set({
      blocks: updated.map((b, i) => ({ ...b, order: i })),
      isDirty: true,
    });
  },

  selectBlock: (id) => set({ selectedBlockId: id }),

  setBlocksFromSuggestion: (blocks) =>
    set({ blocks, selectedBlockId: null, isDirty: true }),

  setTheme: (partial) => {
    const { theme } = get();
    set({ theme: { ...theme, ...partial }, isDirty: true });
  },

  togglePreview: () => set((s) => ({ isPreviewMode: !s.isPreviewMode })),

  markClean: () => set({ isDirty: false }),

  reset: () =>
    set({
      blocks: [],
      selectedBlockId: null,
      theme: { ...DEFAULT_THEME },
      isDirty: false,
      isPreviewMode: false,
    }),
}));
