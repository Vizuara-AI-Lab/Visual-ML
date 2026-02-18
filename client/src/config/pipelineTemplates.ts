/**
 * Pipeline Templates
 *
 * Each template defines a set of pre-connected nodes that get stamped onto
 * the canvas when the user clicks "Load" in the NodePalette Templates tab.
 *
 * Templates to keep:
 *  - Linear Regression  (regression)
 *  - Decision Tree      (classification)
 *  - Random Forest      (classification)
 *  - GenAI Chatbot      (genai)
 */

import type { Node, Edge } from "@xyflow/react";
import type { BaseNodeData, NodeType } from "../types/pipeline";
import { getNodeByType } from "./nodeDefinitions";

// ─── Helpers ─────────────────────────────────────────────────────

/** Build a canvas node from a registered node definition. */
function makeNode(
  type: NodeType,
  pos: { x: number; y: number },
  id: string,
): Node<BaseNodeData> {
  const def = getNodeByType(type);
  if (!def) throw new Error(`Unknown node type: ${type}`);
  return {
    id,
    type,
    position: pos,
    data: {
      label: def.label,
      type: def.type,
      config: JSON.parse(JSON.stringify(def.defaultConfig)),
      isConfigured: false,
      color: def.color,
      icon: def.icon,
    },
  };
}

/** Build a directed pipeline edge. */
function makeEdge(source: string, target: string): Edge {
  return {
    id: `${source}-${target}`,
    source,
    target,
    type: "pipeline",
    animated: true,
    style: { stroke: "#94a3b8", strokeWidth: 2 },
    markerEnd: {
      type: "arrowclosed" as const,
      color: "#94a3b8",
      width: 16,
      height: 16,
    },
  };
}

// ─── Template Type ────────────────────────────────────────────────

export type TemplateCategory = "regression" | "classification" | "genai";

export interface PipelineTemplate {
  id: string;
  name: string;
  description: string;
  category: TemplateCategory;
  /** Short keyword badges shown on the card. */
  tags: string[];
  /** Shown as a badge so the user knows how big the template is. */
  nodeCount: number;
  /** Accent colour used for the card border & icon. */
  color: string;
  /** Emoji icon shown on the card. */
  emoji: string;
  /** Called at load time — ids use Date.now() so multiple loads never collide. */
  buildTemplate: () => { nodes: Node<BaseNodeData>[]; edges: Edge[] };
}

// ─── Template Definitions ─────────────────────────────────────────

const linearRegressionTemplate: PipelineTemplate = {
  id: "linear-regression",
  name: "Linear Regression",
  description:
    "Complete regression pipeline — upload data, preprocess, encode, scale, split, train and evaluate with R², RMSE and MAE.",
  category: "regression",
  tags: ["regression", "beginner"],
  nodeCount: 9,
  color: "#3B82F6",
  emoji: "📈",
  buildTemplate: () => {
    const ts = Date.now();
    const ids = {
      upload:   `upload_file-tpl-${ts}-1`,
      missing:  `missing_value_handler-tpl-${ts}-2`,
      encoding: `encoding-tpl-${ts}-3`,
      scaling:  `scaling-tpl-${ts}-4`,
      split:    `split-tpl-${ts}-5`,
      lr:       `linear_regression-tpl-${ts}-6`,
      r2:       `r2_score-tpl-${ts}-7`,
      rmse:     `rmse_score-tpl-${ts}-8`,
      mae:      `mae_score-tpl-${ts}-9`,
    };

    const X = 220;
    const nodes: Node<BaseNodeData>[] = [
      makeNode("upload_file",           { x: X, y: 50   }, ids.upload),
      makeNode("missing_value_handler", { x: X, y: 200  }, ids.missing),
      makeNode("encoding",              { x: X, y: 350  }, ids.encoding),
      makeNode("scaling",               { x: X, y: 500  }, ids.scaling),
      makeNode("split",                 { x: X, y: 650  }, ids.split),
      makeNode("linear_regression",     { x: X, y: 800  }, ids.lr),
      makeNode("r2_score",              { x: 50,  y: 960 }, ids.r2),
      makeNode("rmse_score",            { x: 220, y: 960 }, ids.rmse),
      makeNode("mae_score",             { x: 390, y: 960 }, ids.mae),
    ];

    const edges: Edge[] = [
      makeEdge(ids.upload,   ids.missing),
      makeEdge(ids.missing,  ids.encoding),
      makeEdge(ids.encoding, ids.scaling),
      makeEdge(ids.scaling,  ids.split),
      makeEdge(ids.split,    ids.lr),
      makeEdge(ids.lr,       ids.r2),
      makeEdge(ids.lr,       ids.rmse),
      makeEdge(ids.lr,       ids.mae),
    ];

    return { nodes, edges };
  },
};

const decisionTreeTemplate: PipelineTemplate = {
  id: "decision-tree",
  name: "Decision Tree",
  description:
    "Classification pipeline — upload data, handle missing values, encode categories, split, train a Decision Tree and inspect the confusion matrix.",
  category: "classification",
  tags: ["classification", "beginner", "tree"],
  nodeCount: 7,
  color: "#10B981",
  emoji: "🌳",
  buildTemplate: () => {
    const ts = Date.now();
    const ids = {
      upload:   `upload_file-tpl-${ts}-1`,
      missing:  `missing_value_handler-tpl-${ts}-2`,
      encoding: `encoding-tpl-${ts}-3`,
      split:    `split-tpl-${ts}-4`,
      dt:       `decision_tree-tpl-${ts}-5`,
      cm:       `confusion_matrix-tpl-${ts}-6`,
      rmse:     `rmse_score-tpl-${ts}-7`,
    };

    const X = 220;
    const nodes: Node<BaseNodeData>[] = [
      makeNode("upload_file",           { x: X, y: 50  }, ids.upload),
      makeNode("missing_value_handler", { x: X, y: 200 }, ids.missing),
      makeNode("encoding",              { x: X, y: 350 }, ids.encoding),
      makeNode("split",                 { x: X, y: 500 }, ids.split),
      makeNode("decision_tree",         { x: X, y: 650 }, ids.dt),
      makeNode("confusion_matrix",      { x: 80,  y: 810 }, ids.cm),
      makeNode("rmse_score",            { x: 360, y: 810 }, ids.rmse),
    ];

    const edges: Edge[] = [
      makeEdge(ids.upload,   ids.missing),
      makeEdge(ids.missing,  ids.encoding),
      makeEdge(ids.encoding, ids.split),
      makeEdge(ids.split,    ids.dt),
      makeEdge(ids.dt,       ids.cm),
      makeEdge(ids.dt,       ids.rmse),
    ];

    return { nodes, edges };
  },
};

const randomForestTemplate: PipelineTemplate = {
  id: "random-forest",
  name: "Random Forest",
  description:
    "Ensemble classification — upload, preprocess, encode, scale, split, train a Random Forest and evaluate with confusion matrix and MAE.",
  category: "classification",
  tags: ["classification", "ensemble", "advanced"],
  nodeCount: 8,
  color: "#8B5CF6",
  emoji: "🌲",
  buildTemplate: () => {
    const ts = Date.now();
    const ids = {
      upload:   `upload_file-tpl-${ts}-1`,
      missing:  `missing_value_handler-tpl-${ts}-2`,
      encoding: `encoding-tpl-${ts}-3`,
      scaling:  `scaling-tpl-${ts}-4`,
      split:    `split-tpl-${ts}-5`,
      rf:       `random_forest-tpl-${ts}-6`,
      cm:       `confusion_matrix-tpl-${ts}-7`,
      mae:      `mae_score-tpl-${ts}-8`,
    };

    const X = 220;
    const nodes: Node<BaseNodeData>[] = [
      makeNode("upload_file",           { x: X, y: 50  }, ids.upload),
      makeNode("missing_value_handler", { x: X, y: 200 }, ids.missing),
      makeNode("encoding",              { x: X, y: 350 }, ids.encoding),
      makeNode("scaling",               { x: X, y: 500 }, ids.scaling),
      makeNode("split",                 { x: X, y: 650 }, ids.split),
      makeNode("random_forest",         { x: X, y: 800 }, ids.rf),
      makeNode("confusion_matrix",      { x: 80,  y: 960 }, ids.cm),
      makeNode("mae_score",             { x: 360, y: 960 }, ids.mae),
    ];

    const edges: Edge[] = [
      makeEdge(ids.upload,   ids.missing),
      makeEdge(ids.missing,  ids.encoding),
      makeEdge(ids.encoding, ids.scaling),
      makeEdge(ids.scaling,  ids.split),
      makeEdge(ids.split,    ids.rf),
      makeEdge(ids.rf,       ids.cm),
      makeEdge(ids.rf,       ids.mae),
    ];

    return { nodes, edges };
  },
};

const genaiChatbotTemplate: PipelineTemplate = {
  id: "genai-chatbot",
  name: "GenAI Chatbot",
  description:
    "AI chat pipeline — wire an LLM Provider and System Prompt into a Chatbot, with Few-Shot Examples for guided responses.",
  category: "genai",
  tags: ["genai", "chatbot", "llm", "few-shot"],
  nodeCount: 4,
  color: "#F59E0B",
  emoji: "🤖",
  buildTemplate: () => {
    const ts = Date.now();
    const ids = {
      llm:      `llm_node-tpl-${ts}-1`,
      system:   `system_prompt-tpl-${ts}-2`,
      examples: `example_node-tpl-${ts}-3`,
      chatbot:  `chatbot_node-tpl-${ts}-4`,
    };

    const nodes: Node<BaseNodeData>[] = [
      makeNode("llm_node",      { x: 50,  y: 120 }, ids.llm),
      makeNode("system_prompt", { x: 50,  y: 290 }, ids.system),
      makeNode("example_node",  { x: 50,  y: 460 }, ids.examples),
      makeNode("chatbot_node",  { x: 420, y: 290 }, ids.chatbot),
    ];

    const edges: Edge[] = [
      makeEdge(ids.llm,      ids.chatbot),
      makeEdge(ids.system,   ids.chatbot),
      makeEdge(ids.examples, ids.chatbot),
    ];

    return { nodes, edges };
  },
};

// ─── Master List ──────────────────────────────────────────────────

export const PIPELINE_TEMPLATES: PipelineTemplate[] = [
  linearRegressionTemplate,
  decisionTreeTemplate,
  randomForestTemplate,
  genaiChatbotTemplate,
];
