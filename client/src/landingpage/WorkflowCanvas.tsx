import React, { useCallback } from "react";
import {
  Database,
  Layers,
  Brain,
  Network,
  FileText,
  BarChart3,
} from "lucide-react";
import {
  ReactFlow,
  type Node,
  type Edge,
  addEdge,
  type Connection,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  BackgroundVariant,
  MarkerType,
  Handle,
  Position,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

// Custom Node Component
const CustomNode = ({
  data,
}: {
  data: { label: string; icon: React.ElementType; description?: string };
}) => {
  const Icon = data.icon;

  return (
    <div className="relative group">
      <Handle
        type="target"
        position={Position.Left}
        className="w-3.5! h-3.5! bg-slate-600! border-2! border-white! shadow-md!"
      />
      <div className="relative bg-white rounded-2xl shadow-xl border border-slate-200/80 p-5 hover:shadow-2xl hover:border-slate-300 transition-all duration-300 min-w-48">
        {/* Subtle gradient overlay */}
        <div className="absolute inset-0 bg-linear-to-br from-slate-50/50 via-transparent to-transparent rounded-2xl pointer-events-none" />

        <div className="relative flex items-start gap-4">
          <div className="relative w-12 h-12 rounded-xl bg-linear-to-br from-slate-900 to-slate-700 flex items-center justify-center shadow-lg shadow-slate-900/25 group-hover:shadow-xl group-hover:shadow-slate-900/30 transition-all flex-shrink-0">
            <Icon className="w-6 h-6 text-white" />
            <div className="absolute inset-0 rounded-xl bg-linear-to-br from-white/10 to-transparent" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="font-semibold text-slate-900 text-base mb-0.5 tracking-tight">
              {data.label}
            </div>
            <div className="text-xs text-slate-500 font-medium">
              {data.description || "Ready"}
            </div>
          </div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Right}
        className="w-3.5! h-3.5! bg-slate-700! border-2! border-white! shadow-md!"
      />
    </div>
  );
};

const nodeTypes = {
  custom: CustomNode,
};

const initialNodes: Node[] = [
  {
    id: "1",
    type: "custom",
    position: { x: 50, y: 165 },
    data: { label: "Data", icon: Database, description: "Load dataset" },
  },
  {
    id: "2",
    type: "custom",
    position: { x: 280, y: 140 },
    data: { label: "Clean", icon: FileText, description: "Preprocess" },
  },
  {
    id: "4",
    type: "custom",
    position: { x: 510, y: 155 },
    data: { label: "Process", icon: Layers, description: "Feature eng." },
  },
  {
    id: "5",
    type: "custom",
    position: { x: 740, y: 70 },
    data: { label: "Train", icon: Brain, description: "Model training" },
  },
  {
    id: "6",
    type: "custom",
    position: { x: 740, y: 225 },
    data: { label: "Validate", icon: BarChart3, description: "Evaluation" },
  },
  {
    id: "7",
    type: "custom",
    position: { x: 970, y: 145 },
    data: { label: "Deploy", icon: Network, description: "Production" },
  },
];

const initialEdges: Edge[] = [
  {
    id: "e1-2",
    source: "1",
    target: "2",
    type: "smoothstep",
    animated: true,
    style: {
      stroke: "url(#edge-gradient)",
      strokeWidth: 3,
      filter: "drop-shadow(0 0 4px rgba(51, 65, 85, 0.3))",
    },
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: "#475569",
      width: 16,
      height: 16,
    },
  },
  {
    id: "e2-4",
    source: "2",
    target: "4",
    type: "smoothstep",
    animated: true,
    style: {
      stroke: "url(#edge-gradient)",
      strokeWidth: 3,
      filter: "drop-shadow(0 0 4px rgba(51, 65, 85, 0.3))",
    },
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: "#475569",
      width: 16,
      height: 16,
    },
  },
  {
    id: "e4-5",
    source: "4",
    target: "5",
    type: "smoothstep",
    animated: true,
    style: {
      stroke: "url(#edge-gradient)",
      strokeWidth: 3,
      filter: "drop-shadow(0 0 4px rgba(51, 65, 85, 0.3))",
    },
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: "#475569",
      width: 16,
      height: 16,
    },
  },
  {
    id: "e4-6",
    source: "4",
    target: "6",
    type: "smoothstep",
    animated: true,
    style: {
      stroke: "url(#edge-gradient)",
      strokeWidth: 3,
      filter: "drop-shadow(0 0 4px rgba(51, 65, 85, 0.3))",
    },
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: "#475569",
      width: 16,
      height: 16,
    },
  },
  {
    id: "e5-7",
    source: "5",
    target: "7",
    type: "smoothstep",
    animated: true,
    style: {
      stroke: "url(#edge-gradient)",
      strokeWidth: 3,
      filter: "drop-shadow(0 0 4px rgba(51, 65, 85, 0.3))",
    },
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: "#475569",
      width: 16,
      height: 16,
    },
  },
  {
    id: "e6-7",
    source: "6",
    target: "7",
    type: "smoothstep",
    animated: true,
    style: {
      stroke: "url(#edge-gradient)",
      strokeWidth: 3,
      filter: "drop-shadow(0 0 4px rgba(51, 65, 85, 0.3))",
    },
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: "#475569",
      width: 16,
      height: 16,
    },
  },
];

const WorkflowCanvas: React.FC = () => {
  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onConnect = useCallback(
    (params: Connection) =>
      setEdges((eds: Edge[]) =>
        addEdge(
          {
            ...params,
            type: "smoothstep",
            animated: true,
            style: {
              stroke: "url(#edge-gradient)",
              strokeWidth: 3,
              filter: "drop-shadow(0 0 4px rgba(51, 65, 85, 0.3))",
            },
            markerEnd: {
              type: MarkerType.ArrowClosed,
              color: "#475569",
              width: 16,
              height: 16,
            },
          },
          eds,
        ),
      ),
    [setEdges],
  );

  const onEdgeClick = useCallback(
    (_event: React.MouseEvent, edge: Edge) => {
      setEdges((eds: Edge[]) => eds.filter((e: Edge) => e.id !== edge.id));
    },
    [setEdges],
  );

  return (
    <>
      <style>{`
        .react-flow__edge-path {
          stroke-width: 3 !important;
          filter: drop-shadow(0 0 4px rgba(51, 65, 85, 0.3));
        }
        .react-flow__edge {
          z-index: 1 !important;
        }
        .react-flow__node {
          z-index: 2 !important;
        }
        .react-flow__handle {
          opacity: 0;
          transition: opacity 0.2s;
        }
        .react-flow__node:hover .react-flow__handle {
          opacity: 1;
        }
      `}</style>
      <svg width="0" height="0">
        <defs>
          <linearGradient id="edge-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop
              offset="0%"
              style={{ stopColor: "#475569", stopOpacity: 1 }}
            />
            <stop
              offset="50%"
              style={{ stopColor: "#64748b", stopOpacity: 1 }}
            />
            <stop
              offset="100%"
              style={{ stopColor: "#475569", stopOpacity: 1 }}
            />
          </linearGradient>
        </defs>
      </svg>
      <div className="relative">
        {/* Glow Effect */}
        <div className="absolute -inset-1 bg-linear-to-r from-blue-600/20 via-violet-600/20 to-pink-600/20 rounded-3xl blur-2xl" />

        {/* Main Dashboard Mockup */}
        <div className="relative bg-white/80 backdrop-blur-xl rounded-3xl shadow-2xl shadow-slate-900/10 border border-white/20 overflow-hidden ring-1 ring-slate-900/5">
          {/* Header */}
          <div className="border-b border-slate-200/80 px-6 py-4 bg-linear-to-b from-slate-50/80 to-white/80 backdrop-blur-xl">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-slate-300 hover:bg-red-400 transition-colors" />
                <div className="w-3 h-3 rounded-full bg-slate-300 hover:bg-yellow-400 transition-colors" />
                <div className="w-3 h-3 rounded-full bg-slate-300 hover:bg-green-400 transition-colors" />
              </div>
              <div className="flex items-center gap-2">
                <div className="text-xs font-mono text-slate-400">
                  ml-pipeline
                </div>
                <div className="text-sm font-semibold text-slate-700">
                  Linear Regression
                </div>
              </div>
              <div className="flex items-center gap-2">
                <div className="px-2 py-1 rounded-md bg-emerald-50 border border-emerald-200">
                  <span className="text-xs font-medium text-emerald-700">
                    Deployed
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Canvas Area */}
          <div className="relative h-125 bg-linear-to-br from-slate-50/50 via-white/50 to-slate-50/50">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onEdgeClick={onEdgeClick}
              nodeTypes={nodeTypes}
              fitView
              proOptions={{ hideAttribution: true }}
              defaultEdgeOptions={{
                type: "smoothstep",
                animated: true,
                style: {
                  stroke: "url(#edge-gradient)",
                  strokeWidth: 3,
                  filter: "drop-shadow(0 0 4px rgba(51, 65, 85, 0.3))",
                },
              }}
            >
              <Background
                variant={BackgroundVariant.Dots}
                gap={20}
                size={1}
                color="#e2e8f0"
                style={{ opacity: 0.3 }}
              />
              <Controls
                className="bg-white/90 backdrop-blur-sm border border-slate-200 rounded-xl shadow-xl"
                showInteractive={false}
              />
            </ReactFlow>
          </div>

          {/* Bottom Status Bar */}
          <div className="border-t border-slate-200/80 px-6 py-3.5 bg-linear-to-b from-white/80 to-slate-50/80 backdrop-blur-xl">
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-5">
                <span className="flex items-center gap-2 text-slate-600">
                  <div className="relative w-2 h-2">
                    <div className="absolute w-2 h-2 bg-emerald-500 rounded-full"></div>
                    <div className="absolute w-2 h-2 bg-emerald-500 rounded-full animate-ping"></div>
                  </div>
                  <span className="font-medium">Active</span>
                </span>
                <span className="text-slate-500">{nodes.length} nodes</span>
                <span className="text-slate-500">
                  {edges.length} connections
                </span>
              </div>
              
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default WorkflowCanvas;
