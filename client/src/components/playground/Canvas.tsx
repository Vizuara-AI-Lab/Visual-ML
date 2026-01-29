import { useCallback, useRef } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type OnConnect,
  type OnNodesChange,
  type OnEdgesChange,
  applyNodeChanges,
  applyEdgeChanges,
  type Node,
  useReactFlow,
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
  llm_node: MLNode,
  prompt_template: MLNode,
  rag_node: MLNode,
  image_generation: MLNode,
  model_export: MLNode,
  api_endpoint: MLNode,
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
          type: "smoothstep",
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
        console.log("❌ No node type in drag data");
        return;
      }

      const nodeDef = getNodeByType(type as NodeType);
      if (!nodeDef) {
        console.log("❌ No node definition found for type:", type);
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

      console.log("✅ Adding node:", newNode);
      console.log("📍 Position:", position);
      console.log("📊 Current nodes count:", nodes.length);
      addNode(newNode);
      console.log("✅ Node added successfully");
    },
    [addNode, screenToFlowPosition, nodes.length],
  );

  return (
    <div ref={reactFlowWrapper} className="flex-1 bg-gray-950">
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
        fitView
        snapToGrid
        snapGrid={[15, 15]}
      >
        <Background color="#374151" gap={16} />
        <Controls />
        <MiniMap
          className="!bg-gray-900 !border-gray-700"
          nodeColor={(node) => {
            const nodeDef = getNodeByType(node.type as NodeType);
            return nodeDef?.color || "#6B7280";
          }}
        />
      </ReactFlow>
    </div>
  );
};
