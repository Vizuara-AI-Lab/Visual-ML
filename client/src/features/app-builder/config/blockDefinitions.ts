/**
 * Block catalog — defines all available block types for the builder palette.
 */

import type {
  BlockType,
  BlockConfig,
  HeroConfig,
  TextConfig,
  FileUploadConfig,
  InputFieldsConfig,
  SubmitButtonConfig,
  ResultsDisplayConfig,
  MetricsCardConfig,
  DividerConfig,
  ImageConfig,
  SpacerConfig,
  AlertConfig,
  CodeConfig,
  VideoEmbedConfig,
} from "../types/appBuilder";

export interface BlockDefinition {
  type: BlockType;
  label: string;
  icon: string; // lucide icon name
  description: string;
  defaultConfig: BlockConfig;
}

export const BLOCK_DEFINITIONS: BlockDefinition[] = [
  {
    type: "hero",
    label: "Hero Header",
    icon: "Layout",
    description: "Page header with title and subtitle",
    defaultConfig: {
      title: "My ML App",
      subtitle: "Powered by machine learning",
      alignment: "center",
      showGradient: true,
      gradientFrom: "#6366f1",
      gradientTo: "#ec4899",
      size: "lg",
      buttonText: "",
      buttonUrl: "",
    } as HeroConfig,
  },
  {
    type: "text",
    label: "Text",
    icon: "Type",
    description: "Paragraph or description text",
    defaultConfig: {
      content: "Enter your text here...",
      size: "md",
      alignment: "left",
    } as TextConfig,
  },
  {
    type: "file_upload",
    label: "File Upload",
    icon: "Upload",
    description: "CSV file upload widget",
    defaultConfig: {
      label: "Upload your data",
      acceptTypes: ".csv",
      maxSizeMB: 10,
      helpText: "Upload a CSV file to run predictions",
    } as FileUploadConfig,
  },
  {
    type: "input_fields",
    label: "Input Fields",
    icon: "FormInput",
    description: "Form inputs for feature values",
    defaultConfig: {
      fields: [
        {
          name: "feature_1",
          label: "Feature 1",
          type: "number",
          placeholder: "Enter value...",
          required: true,
        },
      ],
    } as InputFieldsConfig,
  },
  {
    type: "submit_button",
    label: "Submit Button",
    icon: "Play",
    description: "Execute the pipeline",
    defaultConfig: {
      label: "Run Prediction",
      variant: "gradient",
      loadingText: "Running...",
    } as SubmitButtonConfig,
  },
  {
    type: "results_display",
    label: "Results Display",
    icon: "Table",
    description: "Show pipeline output",
    defaultConfig: {
      title: "Results",
      displayMode: "table",
    } as ResultsDisplayConfig,
  },
  {
    type: "metrics_card",
    label: "Metrics Card",
    icon: "BarChart3",
    description: "Display specific metrics",
    defaultConfig: {
      title: "Model Metrics",
      metrics: [
        { key: "accuracy", label: "Accuracy", format: "percentage" },
      ],
    } as MetricsCardConfig,
  },
  {
    type: "divider",
    label: "Divider",
    icon: "Minus",
    description: "Visual separator",
    defaultConfig: {
      style: "line",
    } as DividerConfig,
  },
  {
    type: "image",
    label: "Image",
    icon: "Image",
    description: "Static image",
    defaultConfig: {
      url: "",
      alt: "Image",
      width: "md",
    } as ImageConfig,
  },
  {
    type: "spacer",
    label: "Spacer",
    icon: "MoveVertical",
    description: "Configurable vertical space",
    defaultConfig: {
      height: 32,
    } as SpacerConfig,
  },
  {
    type: "alert",
    label: "Alert / Notice",
    icon: "AlertTriangle",
    description: "Info, warning, or error notice",
    defaultConfig: {
      variant: "info",
      title: "Note",
      message: "This is an informational notice.",
      showIcon: true,
    } as AlertConfig,
  },
  {
    type: "code",
    label: "Code Block",
    icon: "Code",
    description: "Syntax-highlighted code display",
    defaultConfig: {
      code: "print('Hello, world!')",
      language: "python",
      showLineNumbers: true,
      title: "",
    } as CodeConfig,
  },
  {
    type: "video_embed",
    label: "Video Embed",
    icon: "Video",
    description: "YouTube or Vimeo embed",
    defaultConfig: {
      url: "",
      aspectRatio: "16:9",
      caption: "",
    } as VideoEmbedConfig,
  },
];

export function getBlockDefinition(type: BlockType): BlockDefinition | undefined {
  return BLOCK_DEFINITIONS.find((b) => b.type === type);
}

export function getDefaultConfig(type: BlockType): BlockConfig {
  const def = getBlockDefinition(type);
  if (!def) throw new Error(`Unknown block type: ${type}`);
  return structuredClone(def.defaultConfig);
}
