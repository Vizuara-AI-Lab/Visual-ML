import { useCallback, useRef } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  type OnConnect,
  type OnNodesChange,
  type OnEdgesChange,
  applyNodeChanges,
  applyEdgeChanges,
  type Node,
  useReactFlow,
  type EdgeProps,
  BaseEdge,
  getSmoothStepPath,
  type Edge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { usePlaygroundStore } from "../../store/playgroundStore";
import MLNode from "./MLNode";
import { getNodeByType } from "../../config/nodeDefinitions";
import type { NodeType, BaseNodeData } from "../../types/pipeline";

const nodeTypes: Record<string, React.ComponentType<any>> = {
  upload_file: MLNode,
  select_dataset: MLNode,
  sample_dataset: MLNode,
  table_view: MLNode,
  data_preview: MLNode,
  statistics_view: MLNode,
  column_info: MLNode,
  chart_view: MLNode,
  missing_value_handler: MLNode,
  encoding: MLNode,
  transformation: MLNode,
  split: MLNode,
  preprocess: MLNode,
  feature_selection: MLNode,
  scaling: MLNode,
  train: MLNode,
  evaluate: MLNode,
  // ML Algorithm Nodes
  linear_regression: MLNode,
  logistic_regression: MLNode,
  decision_tree: MLNode,
  random_forest: MLNode,
  // Result & Metrics Nodes
  r2_score: MLNode,
  mse_score: MLNode,
  rmse_score: MLNode,
  mae_score: MLNode,
  confusion_matrix: MLNode,
  classification_report: MLNode,
  accuracy_score: MLNode,
  roc_curve: MLNode,
  feature_importance: MLNode,
  residual_plot: MLNode,
  prediction_table: MLNode,
  // GenAI Nodes
  llm_node: MLNode,
  system_prompt: MLNode,
  chatbot_node: MLNode,
  example_node: MLNode,
  // Activity Nodes
  activity_loss_functions: MLNode,
  activity_linear_regression: MLNode,
  activity_gradient_descent: MLNode,
  activity_logistic_regression: MLNode,
  activity_knn_playground: MLNode,
  activity_kmeans_clustering: MLNode,
  activity_decision_tree: MLNode,
  activity_confusion_matrix: MLNode,
  activity_activation_functions: MLNode,
  activity_neural_network: MLNode,
  activity_backpropagation: MLNode,
  activity_cnn_filters: MLNode,
  activity_overfitting: MLNode,
};

// Custom styled edge with particle animation during execution
const PipelineEdge = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  markerEnd,
}: EdgeProps) => {
  const [edgePath] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    borderRadius: 12,
  });

  const isActive = style?.stroke === "#10B981";

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          stroke: "#94a3b8",
          strokeWidth: 2,
          ...style,
        }}
      />
      {isActive && (
        <>
          {/* Glow effect */}
          <path
            d={edgePath}
            fill="none"
            stroke="#10B981"
            strokeWidth={6}
            strokeOpacity={0.15}
            style={{ pointerEvents: "none" }}
          />
          {/* Flowing particles */}
          {[0, 0.33, 0.66].map((offset) => (
            <circle key={offset} r={3} fill="#10B981" opacity={0.9}>
              <animateMotion
                dur="1.5s"
                repeatCount="indefinite"
                begin={`${offset * 1.5}s`}
                path={edgePath}
              />
            </circle>
          ))}
        </>
      )}
    </>
  );
};

const edgeTypes = {
  pipeline: PipelineEdge,
};

const defaultEdgeOptions = {
  type: "pipeline" as const,
  animated: true,
  style: { stroke: "#94a3b8", strokeWidth: 2 },
  markerEnd: {
    type: "arrowclosed" as const,
    color: "#94a3b8",
    width: 16,
    height: 16,
  },
};

interface CanvasProps {
  onNodeClick: (nodeId: string) => void;
}

export const Canvas = ({ onNodeClick }: CanvasProps) => {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition } = useReactFlow();
  const { nodes, edges, addNode, setNodes, setEdges, addEdge } =
    usePlaygroundStore();

  const onNodesChange: OnNodesChange = useCallback(
    (changes) => {
      setNodes(
        (prevNodes) =>
          applyNodeChanges(changes, prevNodes) as Node<BaseNodeData>[],
      );
    },
    [setNodes], // Stable dependency - prevents unnecessary re-renders
  );

  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => {
      setEdges((prevEdges) => applyEdgeChanges(changes, prevEdges));
    },
    [setEdges], // Stable dependency
  );

  const onConnect: OnConnect = useCallback(
    (connection) => {
      if (connection.source && connection.target) {
        addEdge({
          id: `${connection.source}-${connection.target}`,
          source: connection.source,
          target: connection.target,
          type: "pipeline",
          animated: true,
        });
      }
    },
    [addEdge],
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      event.stopPropagation();

      const type = event.dataTransfer.getData("application/reactflow");
      if (!type) {
        return;
      }

      const nodeDef = getNodeByType(type as NodeType);
      if (!nodeDef) {
        return;
      }

      // Convert screen coordinates to flow coordinates
      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const newNode = {
        id: `${type}-${Date.now()}`,
        type,
        position,
        data: {
          label: nodeDef.label,
          type: nodeDef.type,
          // Deep clone defaultConfig so each instance has independent state
          config: JSON.parse(JSON.stringify(nodeDef.defaultConfig)),
          isConfigured: false,
          color: nodeDef.color,
          icon: nodeDef.icon, // Store the icon component or string directly
        },
      };

      addNode(newNode);
    },
    [addNode, screenToFlowPosition, nodes.length],
  );

  return (
    <div ref={reactFlowWrapper} className="flex-1 bg-slate-50">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onNodeClick={(_, node) => onNodeClick(node.id)}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        defaultEdgeOptions={defaultEdgeOptions}
        fitView
        snapToGrid
        snapGrid={[15, 15]}
      >
        <Background color="#cbd5e1" gap={20} size={1.5} />
        <Controls className="rounded-xl! border-slate-200! shadow-md! [&>button]:rounded-lg! [&>button]:border-slate-200!" />
      </ReactFlow>
    </div>
  );
};
