/**
 * Type definitions for ML Pipeline nodes and configuration
 */

export type NodeType =
  | "upload_file"
  | "select_dataset"
  | "sample_dataset"
  | "table_view"
  | "data_preview"
  | "statistics_view"
  | "column_info"
  | "chart_view"
  | "preprocess"
  | "missing_value_handler"
  | "encoding"
  | "split"
  | "feature_selection"
  | "scaling"
  | "linear_regression"
  | "logistic_regression"
  | "decision_tree"
  | "random_forest"
  | "mlp_classifier"
  | "mlp_regressor"
  // Result & Metrics nodes
  | "r2_score"
  | "mse_score"
  | "rmse_score"
  | "mae_score"
  | "confusion_matrix"
  // GenAI nodes
  | "llm_node"
  | "system_prompt"
  | "chatbot_node"
  | "example_node"
  // Image pipeline nodes
  | "image_dataset"
  | "image_preprocessing"
  | "image_split"
  | "cnn_classifier"
  | "image_predictions"
  // Activity nodes (standalone interactive learning)
  | "activity_loss_functions"
  | "activity_linear_regression"
  | "activity_gradient_descent"
  | "activity_logistic_regression"
  | "activity_kmeans_clustering"
  | "activity_decision_tree"
  | "activity_confusion_matrix"
  | "activity_activation_functions"
  | "activity_neural_network"
  | "activity_backpropagation"
  | "activity_cnn_filters";

export type AlgorithmType = "linear_regression" | "logistic_regression" | "mlp_classifier" | "mlp_regressor";

export type TaskType = "regression" | "classification";

export interface BaseNodeData extends Record<string, unknown> {
  label: string;
  type: NodeType;
  config: Record<string, any>;
  isConfigured: boolean;
  validationErrors?: string[];
  icon?: string | React.ComponentType<any>;
  color?: string;
}

// Preprocess Node Configuration
export interface PreprocessConfig {
  dataset_path: string;
  target_column: string;
  handle_missing: boolean;
  missing_strategy: "drop" | "mean" | "median" | "mode" | "fill";
  fill_value?: number;
  text_columns: string[];
  numeric_columns: string[];
  lowercase: boolean;
  remove_stopwords: boolean;
  max_features: number;
  scale_features: boolean;
}

// Split Node Configuration
export interface SplitConfig {
  dataset_path: string;
  target_column: string;
  train_ratio: number;
  val_ratio?: number;
  test_ratio: number;
  random_seed: number;
  shuffle: boolean;
  stratify: boolean;
}

// Train Node Configuration
export interface TrainConfig {
  train_dataset_path: string;
  target_column: string;
  algorithm: AlgorithmType;
  task_type: TaskType;
  hyperparameters: Record<string, any>;
  model_name?: string;
}

// Evaluate Node Configuration
export interface EvaluateConfig {
  model_path: string;
  test_dataset_path: string;
  target_column: string;
  task_type: TaskType;
}

// Node execution request/response types
export interface NodeExecuteRequest {
  node_type: NodeType;
  input_data: Record<string, any>;
  node_id?: string;
  dry_run?: boolean;
}

export interface NodeExecuteResponse {
  success: boolean;
  node_type: NodeType;
  result?: Record<string, any>;
  error?: Record<string, any>;
}

// Pipeline execution types
export interface PipelineNodeConfig {
  node_type: NodeType;
  input: Record<string, any>;
  node_id?: string;
}

export interface EdgeInfo {
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
}

export interface PipelineExecuteRequest {
  pipeline: PipelineNodeConfig[];
  edges?: EdgeInfo[];
  dry_run?: boolean;
  pipeline_name?: string;
}

export interface PipelineExecuteResponse {
  success: boolean;
  pipeline_name?: string;
  results: Array<Record<string, any>>;
  total_execution_time_seconds: number;
}

// Hyperparameters for different algorithms
export interface LinearRegressionHyperparameters {
  fit_intercept?: boolean;
  copy_X?: boolean;
  n_jobs?: number;
}

export interface LogisticRegressionHyperparameters {
  C?: number;
  penalty?: "l1" | "l2" | "elasticnet" | "none";
  solver?: "lbfgs" | "liblinear" | "newton-cg" | "sag" | "saga";
  max_iter?: number;
  tol?: number;
  fit_intercept?: boolean;
  class_weight?: "balanced" | null;
  random_state?: number;
  n_jobs?: number;
}

// Node metadata for palette
export interface NodeMetadata {
  type: NodeType;
  label: string;
  description: string;
  color: string;
  icon: string;
  category: "input" | "preprocessing" | "model" | "output";
  defaultConfig: Record<string, any>;
}

// SSE streaming event types
export type NodeExecutionStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed";

export interface NodeStartedEvent {
  event: "node_started";
  node_id: string;
  node_type: NodeType;
  label: string;
}

export interface NodeCompletedEvent {
  event: "node_completed";
  node_id: string;
  node_type: NodeType;
  label: string;
  success: true;
  result?: Record<string, any>;
}

export interface NodeFailedEvent {
  event: "node_failed";
  node_id: string;
  node_type: NodeType;
  label: string;
  error: string;
}

export interface PipelineCompletedEvent {
  event: "pipeline_completed";
  success: true;
  nodes_executed: number;
  execution_time_seconds: number;
  results: Array<Record<string, any>>;
}

export interface PipelineFailedEvent {
  event: "pipeline_failed";
  success: false;
  error: string;
}

export type PipelineStreamEvent =
  | NodeStartedEvent
  | NodeCompletedEvent
  | NodeFailedEvent
  | PipelineCompletedEvent
  | PipelineFailedEvent;

export interface PipelineStreamCallbacks {
  onNodeStarted?: (event: NodeStartedEvent) => void;
  onNodeCompleted?: (event: NodeCompletedEvent) => void;
  onNodeFailed?: (event: NodeFailedEvent) => void;
  onPipelineCompleted?: (event: PipelineCompletedEvent) => void;
  onPipelineFailed?: (event: PipelineFailedEvent) => void;
  onError?: (error: Error) => void;
}

// Execution log entry for the results drawer timeline
export interface ExecutionLogEntry {
  timestamp: string;
  event:
    | "node_started"
    | "node_completed"
    | "node_failed"
    | "pipeline_completed"
    | "pipeline_failed";
  nodeId?: string;
  nodeLabel?: string;
  message: string;
}
