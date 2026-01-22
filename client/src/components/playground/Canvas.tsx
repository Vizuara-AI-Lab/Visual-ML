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
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { usePlaygroundStore } from "../../store/playgroundStore";
import MLNode from "./MLNode";
import { getNodeByType } from "../../config/nodeDefinitions";
import type { NodeType, BaseNodeData } from "../../types/pipeline";

const nodeTypes: Record<string, React.ComponentType<any>> = {
  upload_file: MLNode,
  load_url: MLNode,
  sample_dataset: MLNode,
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
  const { nodes, edges, addNode, setNodes, setEdges, addEdge } =
    usePlaygroundStore();

  const onNodesChange: OnNodesChange = useCallback(
    (changes) => {
      setNodes(applyNodeChanges(changes, nodes) as Node<BaseNodeData>[]);
    },
    [setNodes, nodes],
  );

  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => {
      setEdges(applyEdgeChanges(changes, edges));
    },
    [setEdges, edges],
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

      const reactFlowBounds = reactFlowWrapper.current?.getBoundingClientRect();
      if (!reactFlowBounds) return;

      const type = event.dataTransfer.getData("application/reactflow");
      if (!type) return;

      const nodeDef = getNodeByType(type as NodeType);
      if (!nodeDef) return;

      const position = {
        x: event.clientX - reactFlowBounds.left - 100,
        y: event.clientY - reactFlowBounds.top - 25,
      };

      const newNode = {
        id: `${type}-${Date.now()}`,
        type,
        position,
        data: {
          label: nodeDef.label,
          type: nodeDef.type,
          config: nodeDef.defaultConfig,
          isConfigured: false,
          color: nodeDef.color,
          icon: nodeDef.icon.name,
        },
      };

      addNode(newNode);
    },
    [addNode],
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
