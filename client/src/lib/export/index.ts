/**
 * Export-to-Code public API.
 *
 * Usage:
 *   import { exportToPython, exportToNotebook } from "@/lib/export";
 *
 *   const { pythonSource } = exportToPython(nodes, edges);
 *   downloadFile("pipeline.py", pythonSource);
 *
 *   const notebookJson = exportToNotebook(nodes, edges);
 *   downloadFile("pipeline.ipynb", notebookJson);
 */

import type { Node, Edge } from "@xyflow/react";
import type { BaseNodeData } from "../../types/pipeline";
import type { GeneratedPipeline, ExportFormat } from "./types";
import { generatePipeline } from "./codeGenerator";
import { toNotebook, serializeNotebook } from "./notebookFormatter";

export type { GeneratedPipeline, ExportFormat };

/** Generate a Python pipeline from the canvas graph */
export function exportToPython(
  nodes: Node<BaseNodeData>[],
  edges: Edge[],
): GeneratedPipeline {
  return generatePipeline(nodes, edges);
}

/** Generate a Jupyter Notebook JSON string from the canvas graph */
export function exportToNotebook(
  nodes: Node<BaseNodeData>[],
  edges: Edge[],
): string {
  const pipeline = generatePipeline(nodes, edges);
  const doc = toNotebook(pipeline);
  return serializeNotebook(doc);
}

/** Trigger a file download in the browser */
export function downloadFile(
  filename: string,
  content: string,
  mimeType = "text/plain",
): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/** Copy text to clipboard, returns true on success */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}
