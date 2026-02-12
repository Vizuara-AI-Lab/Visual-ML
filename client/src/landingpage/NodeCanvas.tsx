import React, { useState, useEffect, useRef } from "react";
import {
  Database,
  Sparkles,
  Layers,
  Cpu,
  BarChart3,
  Rocket,
  Zap,
} from "lucide-react";

interface Node {
  id: string;
  label: string;
  icon: React.ElementType;
  x: number;
  y: number;
}

interface Connection {
  from: string;
  to: string;
}

interface FlowParticle {
  id: string;
  connectionIndex: number;
  progress: number;
}

const NodeCanvas: React.FC = () => {
  const [nodes, setNodes] = useState<Node[]>([
    { id: "1", label: "Dataset", icon: Database, x: 100, y: 80 },
    { id: "2", label: "Clean", icon: Sparkles, x: 400, y: 80 },
    { id: "3", label: "Embed", icon: Layers, x: 700, y: 80 },
    { id: "4", label: "Train", icon: Cpu, x: 250, y: 250 },
    { id: "5", label: "Evaluate", icon: BarChart3, x: 550, y: 250 },
    { id: "6", label: "GenAI Prompt", icon: Zap, x: 400, y: 420 },
    { id: "7", label: "Deploy", icon: Rocket, x: 700, y: 420 },
  ]);

  const connections: Connection[] = [
    { from: "1", to: "2" },
    { from: "2", to: "3" },
    { from: "1", to: "4" },
    { from: "3", to: "5" },
    { from: "4", to: "5" },
    { from: "5", to: "6" },
    { from: "6", to: "7" },
  ];

  const [dragging, setDragging] = useState<string | null>(null);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [flowParticles, setFlowParticles] = useState<FlowParticle[]>([]);
  const canvasRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const particles: FlowParticle[] = connections.map((_, idx) => ({
      id: `particle-${idx}`,
      connectionIndex: idx,
      progress: Math.random(),
    }));
    setFlowParticles(particles);

    const interval = setInterval(() => {
      setFlowParticles((prev) =>
        prev.map((p) => ({
          ...p,
          progress: (p.progress + 0.01) % 1,
        })),
      );
    }, 50);

    return () => clearInterval(interval);
  }, []);

  const palette = [
    { label: "Dataset", icon: Database },
    { label: "Clean", icon: Sparkles },
    { label: "GenAI", icon: Zap },
    { label: "Deploy", icon: Rocket },
  ];

  const handlePointerDown = (e: React.PointerEvent, nodeId: string) => {
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return;

    const rect = e.currentTarget.getBoundingClientRect();
    setOffset({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
    setDragging(nodeId);
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!dragging) return;

    const canvas = e.currentTarget.getBoundingClientRect();
    const newX = e.clientX - canvas.left - offset.x;
    const newY = e.clientY - canvas.top - offset.y;

    const clampedX = Math.max(0, Math.min(newX, canvas.width - 120));
    const clampedY = Math.max(0, Math.min(newY, canvas.height - 60));

    setNodes((prev) =>
      prev.map((node) =>
        node.id === dragging ? { ...node, x: clampedX, y: clampedY } : node,
      ),
    );
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    setDragging(null);
    (e.target as HTMLElement).releasePointerCapture(e.pointerId);
  };

  const getNodeCenter = (nodeId: string) => {
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return { x: 0, y: 0 };
    return { x: node.x + 60, y: node.y + 24 };
  };

  const drawConnection = (from: string, to: string) => {
    const start = getNodeCenter(from);
    const end = getNodeCenter(to);

    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const distance = Math.sqrt(dx * dx + dy * dy);

    return {
      x1: start.x,
      y1: start.y,
      x2: end.x,
      y2: end.y,
      distance,
    };
  };

  return (
    <div className="absolute inset-0 w-full h-full bg-white/50 backdrop-blur-sm rounded-2xl border border-gray-200 shadow-xl overflow-hidden">
      <div className="absolute top-6 left-6 z-10 flex gap-2">
        {palette.map((item, idx) => (
          <div
            key={idx}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-white/90 backdrop-blur-sm rounded-lg shadow-sm text-xs font-medium text-gray-700"
          >
            <item.icon className="w-3.5 h-3.5" />
            <span>{item.label}</span>
          </div>
        ))}
      </div>

      <div
        ref={canvasRef}
        className="relative w-full h-full overflow-hidden"
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
      >
        <svg className="absolute inset-0 w-full h-full pointer-events-none">
          {connections.map((connection, idx) => {
            const { x1, y1, x2, y2 } = drawConnection(
              connection.from,
              connection.to,
            );
            const particle = flowParticles.find(
              (p) => p.connectionIndex === idx,
            );

            let particleX = 0;
            let particleY = 0;
            if (particle) {
              particleX = x1 + (x2 - x1) * particle.progress;
              particleY = y1 + (y2 - y1) * particle.progress;
            }

            return (
              <g key={idx}>
                <line
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke="#d1d5db"
                  strokeWidth="2"
                  strokeDasharray="4 4"
                />
                {particle && (
                  <circle
                    cx={particleX}
                    cy={particleY}
                    r="4"
                    fill="#374151"
                    opacity="0.8"
                  />
                )}
              </g>
            );
          })}
        </svg>

        {nodes.map((node) => {
          const Icon = node.icon;
          return (
            <div
              key={node.id}
              className="absolute cursor-move select-none z-20"
              style={{
                left: `${node.x}px`,
                top: `${node.y}px`,
                transform: dragging === node.id ? "scale(1.05)" : "scale(1)",
                transition: dragging === node.id ? "none" : "transform 0.2s",
              }}
              onPointerDown={(e) => handlePointerDown(e, node.id)}
            >
              <div className="bg-white/95 backdrop-blur-sm border border-gray-300 rounded-lg px-4 py-3 shadow-lg hover:shadow-xl transition-shadow min-w-[120px]">
                <div className="flex items-center gap-2">
                  <div className="flex-shrink-0 w-8 h-8 bg-gray-100 rounded-md flex items-center justify-center">
                    <Icon className="w-4 h-4 text-gray-700" />
                  </div>
                  <span className="text-sm font-medium text-gray-900">
                    {node.label}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default NodeCanvas;
