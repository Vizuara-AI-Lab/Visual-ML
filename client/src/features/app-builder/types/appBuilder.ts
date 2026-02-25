/**
 * Custom App Builder — TypeScript Types
 */

// ─── Block Types ──────────────────────────────────────────────────

export type BlockType =
  | "hero"
  | "text"
  | "file_upload"
  | "input_fields"
  | "submit_button"
  | "results_display"
  | "metrics_card"
  | "divider"
  | "image"
  | "spacer"
  | "alert"
  | "code"
  | "video_embed";

export interface HeroConfig {
  title: string;
  subtitle: string;
  alignment: "left" | "center" | "right";
  showGradient: boolean;
  gradientFrom: string;
  gradientTo: string;
  size: "sm" | "md" | "lg" | "xl";
  buttonText: string;
  buttonUrl: string;
}

export interface TextConfig {
  content: string;
  size: "sm" | "md" | "lg";
  alignment: "left" | "center" | "right";
}

export interface FileUploadConfig {
  label: string;
  acceptTypes: string;
  maxSizeMB: number;
  helpText: string;
  nodeId?: string; // mapped upload_file node
}

export interface InputField {
  name: string;
  label: string;
  type: "text" | "number" | "select" | "textarea";
  placeholder: string;
  required: boolean;
  options?: string[]; // for select type
}

export interface FieldMapping {
  fieldName: string;
  nodeId: string;
  nodeConfigKey: string;
}

export interface InputFieldsConfig {
  fields: InputField[];
  fieldMappings?: FieldMapping[];
}

export interface SubmitButtonConfig {
  label: string;
  variant: "primary" | "secondary" | "gradient";
  loadingText: string;
}

export interface ResultsDisplayConfig {
  title: string;
  displayMode: "table" | "card" | "json";
  nodeId?: string; // specific node's output to display
}

export interface MetricItem {
  key: string;
  label: string;
  format: "number" | "percentage" | "text";
  nodeId?: string;
  nodeOutputKey?: string;
}

export interface MetricsCardConfig {
  title: string;
  metrics: MetricItem[];
}

export interface DividerConfig {
  style: "line" | "space" | "dots";
}

export interface ImageConfig {
  url: string;
  base64Data?: string;
  alt: string;
  width: "sm" | "md" | "lg" | "full";
}

export interface SpacerConfig {
  height: number;
}

export interface AlertConfig {
  variant: "info" | "warning" | "success" | "error";
  title: string;
  message: string;
  showIcon: boolean;
}

export interface CodeConfig {
  code: string;
  language: string;
  showLineNumbers: boolean;
  title: string;
}

export interface VideoEmbedConfig {
  url: string;
  aspectRatio: "16:9" | "4:3";
  caption: string;
}

export type BlockConfig =
  | HeroConfig
  | TextConfig
  | FileUploadConfig
  | InputFieldsConfig
  | SubmitButtonConfig
  | ResultsDisplayConfig
  | MetricsCardConfig
  | DividerConfig
  | ImageConfig
  | SpacerConfig
  | AlertConfig
  | CodeConfig
  | VideoEmbedConfig;

export interface BlockStyleConfig {
  backgroundColor?: string;
  textColor?: string;
  borderRadius?: number;
  paddingX?: number;
  paddingY?: number;
  borderColor?: string;
  borderWidth?: number;
}

export interface AppBlock {
  id: string;
  type: BlockType;
  config: BlockConfig;
  style?: BlockStyleConfig;
  order: number;
  nodeId?: string;    // mapped pipeline node ID
  nodeType?: string;  // mapped pipeline node type
  nodeLabel?: string; // human-readable node label
}

// ─── Theme ────────────────────────────────────────────────────────

export interface AppTheme {
  primaryColor: string;
  fontFamily: string;
  darkMode: boolean;
}

// ─── App ──────────────────────────────────────────────────────────

export interface CustomApp {
  id: number;
  studentId: number;
  pipelineId: number;
  name: string;
  slug: string;
  description: string | null;
  blocks: AppBlock[];
  theme: AppTheme | null;
  is_published: boolean;
  published_at: string | null;
  view_count: number;
  execution_count: number;
  createdAt: string;
  updatedAt: string;
}

// ─── API Request/Response Types ───────────────────────────────────

export interface CreateAppRequest {
  pipeline_id: number;
  name: string;
}

export interface UpdateAppRequest {
  name?: string;
  description?: string;
  blocks?: AppBlock[];
  theme?: AppTheme;
  slug?: string;
}

export interface PublishAppRequest {
  is_published: boolean;
  slug?: string;
}

export interface PublicApp {
  name: string;
  description: string | null;
  blocks: AppBlock[];
  theme: AppTheme | null;
  owner_name: string;
}

export interface ExecuteAppRequest {
  input_data: Record<string, unknown>;
  file_data?: string; // base64
  node_inputs?: Record<string, Record<string, unknown>>; // { nodeId: { configKey: value } }
  file_node_id?: string; // which upload node receives the file
}

export interface ExecuteAppResponse {
  success: boolean;
  results?: Record<string, unknown>;
  error?: string;
  execution_time_ms?: number;
}

export interface SlugCheckResponse {
  available: boolean;
  slug: string;
}

export interface SuggestedBlocksResponse {
  blocks: AppBlock[];
  pipeline_name: string;
  node_count: number;
}
