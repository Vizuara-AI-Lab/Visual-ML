/**
 * Node Palette - Draggable nodes for the pipeline builder
 */

import { nodeDefinitions } from "../../lib/nodeDefinitions";

const NodePalette = () => {
  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData("application/reactflow", nodeType);
    event.dataTransfer.effectAllowed = "move";
  };

  const categories = [
    { id: "input", label: "Data Input" },
    { id: "preprocessing", label: "Preprocessing" },
    { id: "model", label: "Model Training" },
    { id: "output", label: "Results" },
  ];

  return (
    <div className="w-64 bg-white border-r border-gray-200 p-4 overflow-y-auto">
      <h2 className="text-lg font-bold text-gray-800 mb-4">ML Nodes</h2>

      {categories.map((category) => {
        const nodes = nodeDefinitions.filter(
          (node) => node.category === category.id,
        );
        if (nodes.length === 0) return null;

        return (
          <div key={category.id} className="mb-6">
            <h3 className="text-sm font-semibold text-gray-600 mb-2">
              {category.label}
            </h3>
            <div className="space-y-2">
              {nodes.map((node) => (
                <div
                  key={node.type}
                  draggable
                  onDragStart={(e) => onDragStart(e, node.type)}
                  className="p-3 bg-gray-50 rounded-lg border border-gray-200 cursor-move hover:bg-gray-100 hover:shadow-md transition-all"
                  style={{
                    borderLeftWidth: "3px",
                    borderLeftColor: node.color,
                  }}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-lg">{node.icon}</span>
                    <div className="flex-1">
                      <div className="text-sm font-medium text-gray-800">
                        {node.label}
                      </div>
                      <div className="text-xs text-gray-500">
                        {node.description}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      })}

      <div className="mt-6 p-3 bg-blue-50 rounded-lg border border-blue-200">
        <p className="text-xs text-blue-800">
          <strong>Tip:</strong> Drag nodes to the canvas and connect them to
          build your ML pipeline.
        </p>
      </div>
    </div>
  );
};

export default NodePalette;
