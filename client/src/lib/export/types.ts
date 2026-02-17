/**
 * Types for the Export-to-Code feature.
 */

import type { NodeType } from "../../types/pipeline";

/** A single block of generated code from one node */
export interface CodeBlock {
  /** Python import statements needed by this block */
  imports: string[];
  /** The Python code body */
  code: string;
  /** Markdown comment/header for notebook cells */
  comment?: string;
}

/** Function signature for a node code template */
export type TemplateFunction = (
  config: Record<string, unknown>,
  outputVar: string,
  inputVar: string | null,
) => CodeBlock;

/** The complete generated pipeline */
export interface GeneratedPipeline {
  /** All import statements, deduplicated */
  imports: string[];
  /** Ordered code blocks (one per node) */
  blocks: CodeBlock[];
  /** The full assembled Python source */
  pythonSource: string;
}

/** Export format options */
export type ExportFormat = "python" | "notebook";

/** A Jupyter notebook cell */
export interface NotebookCell {
  cell_type: "code" | "markdown";
  source: string[];
  metadata: Record<string, unknown>;
  outputs?: unknown[];
  execution_count?: number | null;
}

/** A Jupyter notebook document */
export interface NotebookDocument {
  nbformat: 4;
  nbformat_minor: 5;
  metadata: {
    kernelspec: {
      display_name: string;
      language: string;
      name: string;
    };
    language_info: {
      name: string;
      version: string;
    };
  };
  cells: NotebookCell[];
}

/** Mapping of node types to variable name prefixes */
export const VARIABLE_PREFIXES: Partial<Record<NodeType, string>> = {
  upload_file: "df",
  select_dataset: "df",
  sample_dataset: "df",
  missing_value_handler: "df_clean",
  preprocess: "df_clean",
  encoding: "df_encoded",
  scaling: "df_scaled",
  transformation: "df_transformed",
  feature_selection: "df_selected",
  split: "split",
  linear_regression: "model",
  logistic_regression: "model",
  decision_tree: "model",
  random_forest: "model",
};
