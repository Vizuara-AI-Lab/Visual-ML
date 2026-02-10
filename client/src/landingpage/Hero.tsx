import React, { useState, useCallback } from "react";
import {
  Sparkles,
  Workflow,
  Share2,
  Database,
  Layers,
  Cpu,
  BarChart3,
  Rocket,
  Zap,
} from "lucide-react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router";
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  type Node,
  type Edge,
  type NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

// Custom Node Component
const CustomNode = ({
  data,
}: {
  data: { label: string; icon: React.ElementType };
}) => {
  const Icon = data.icon;
  return (
    <div className="bg-white/95 backdrop-blur-sm border border-gray-300 rounded-lg px-4 py-3 shadow-lg hover:shadow-xl transition-shadow min-w-[120px] cursor-move">
      <div className="flex items-center gap-2">
        <div className="flex-shrink-0 w-8 h-8 bg-gray-100 rounded-md flex items-center justify-center">
          <Icon className="w-4 h-4 text-gray-700" />
        </div>
        <span className="text-sm font-medium text-gray-900">{data.label}</span>
      </div>
    </div>
  );
};

const nodeTypes: NodeTypes = {
  custom: CustomNode,
};

const Hero: React.FC = () => {
  const navigate = useNavigate();

  const highlights = [
    { icon: Sparkles, text: "GenAI workflows" },
    { icon: Workflow, text: "Custom UI builder" },
    { icon: Share2, text: "Shareable pipelines" },
  ];

  const initialNodes: Node[] = [
    {
      id: "1",
      type: "custom",
      position: { x: 100, y: 80 },
      data: { label: "Dataset", icon: Database },
    },
    {
      id: "2",
      type: "custom",
      position: { x: 400, y: 80 },
      data: { label: "Clean", icon: Sparkles },
    },
    {
      id: "3",
      type: "custom",
      position: { x: 700, y: 80 },
      data: { label: "Embed", icon: Layers },
    },
    {
      id: "4",
      type: "custom",
      position: { x: 250, y: 250 },
      data: { label: "Train", icon: Cpu },
    },
    {
      id: "5",
      type: "custom",
      position: { x: 550, y: 250 },
      data: { label: "Evaluate", icon: BarChart3 },
    },
    {
      id: "6",
      type: "custom",
      position: { x: 400, y: 420 },
      data: { label: "GenAI Prompt", icon: Zap },
    },
    {
      id: "7",
      type: "custom",
      position: { x: 700, y: 420 },
      data: { label: "Deploy", icon: Rocket },
    },
  ];

  const initialEdges: Edge[] = [
    {
      id: "e1-2",
      source: "1",
      target: "2",
      animated: true,
      style: { stroke: "#d1d5db" },
    },
    {
      id: "e2-3",
      source: "2",
      target: "3",
      animated: true,
      style: { stroke: "#d1d5db" },
    },
    {
      id: "e1-4",
      source: "1",
      target: "4",
      animated: true,
      style: { stroke: "#d1d5db" },
    },
    {
      id: "e3-5",
      source: "3",
      target: "5",
      animated: true,
      style: { stroke: "#d1d5db" },
    },
    {
      id: "e4-5",
      source: "4",
      target: "5",
      animated: true,
      style: { stroke: "#d1d5db" },
    },
    {
      id: "e5-6",
      source: "5",
      target: "6",
      animated: true,
      style: { stroke: "#d1d5db" },
    },
    {
      id: "e6-7",
      source: "6",
      target: "7",
      animated: true,
      style: { stroke: "#d1d5db" },
    },
  ];

  const [nodes, setNodes] = useState<Node[]>(initialNodes);
  const [edges, setEdges] = useState<Edge[]>(initialEdges);

  const onNodesChange = useCallback((changes: any) => {
    setNodes((nds) => {
      const updatedNodes = [...nds];
      changes.forEach((change: any) => {
        if (change.type === "position" && change.position) {
          const nodeIndex = updatedNodes.findIndex((n) => n.id === change.id);
          if (nodeIndex !== -1) {
            updatedNodes[nodeIndex] = {
              ...updatedNodes[nodeIndex],
              position: change.position,
            };
          }
        }
      });
      return updatedNodes;
    });
  }, []);

  return (
    <section className="relative h-screen overflow-hidden">
      {/* ReactFlow Canvas with Dotted Background */}
      <div className="absolute inset-0 w-full h-full">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          nodeTypes={nodeTypes}
          fitView
          attributionPosition="bottom-left"
          nodesDraggable={true}
          nodesConnectable={false}
          elementsSelectable={true}
        >
          <Background
            color="#94a3b8"
            variant={BackgroundVariant.Dots}
            gap={16}
            size={1}
          />
        </ReactFlow>
      </div>

      {/* Overlay Content - Text on Left */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="max-w-7xl mx-auto px-6 lg:px-8 h-full flex items-center">
          <motion.div
            className="w-full lg:w-1/2 space-y-8 pointer-events-auto"
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          >
            <div className="space-y-6">
              <motion.h1
                className="text-5xl lg:text-6xl font-bold text-gray-900 leading-tight"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
              >
                Build ML workflows visually.
              </motion.h1>

              <motion.p
                className="text-xl text-gray-600 leading-relaxed"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.4 }}
              >
                Create powerful machine learning pipelines without code.
                Leverage GenAI assistance to build, validate, and deploy
                workflows. Export custom UIs and share your work instantly.
              </motion.p>
            </div>

            <motion.div
              className="flex flex-wrap gap-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
            >
              <motion.button
                onClick={() => navigate("/signup")}
                className="px-6 py-3 bg-gray-900 text-white font-medium rounded-lg hover:bg-gray-800 transition-colors shadow-sm hover:shadow-lg"
                whileHover={{
                  scale: 1.05,
                  boxShadow: "0 10px 30px rgba(0,0,0,0.2)",
                }}
                whileTap={{ scale: 0.95 }}
              >
                Try the builder
              </motion.button>
            </motion.div>

            <motion.div
              className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.8 }}
            >
              {highlights.map((item, index) => (
                <motion.div
                  key={index}
                  className="flex items-center space-x-3"
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.4, delay: 0.9 + index * 0.1 }}
                  whileHover={{ scale: 1.05 }}
                >
                  <div className="flex-shrink-0 w-10 h-10 bg-white/80 backdrop-blur-sm rounded-lg flex items-center justify-center shadow-sm">
                    <item.icon className="w-5 h-5 text-gray-900" />
                  </div>
                  <span className="text-sm font-medium text-gray-900">
                    {item.text}
                  </span>
                </motion.div>
              ))}
            </motion.div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
