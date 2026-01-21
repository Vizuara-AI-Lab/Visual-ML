import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import {
  nodeCategories,
  type NodeDefinition,
} from "../../config/nodeDefinitions";

interface SidebarProps {
  onNodeDragStart: (event: React.DragEvent, nodeType: string) => void;
}

export const Sidebar = ({ onNodeDragStart }: SidebarProps) => {
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(["data-sources", "ml-algorithms"]),
  );

  const toggleCategory = (categoryId: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(categoryId)) {
        next.delete(categoryId);
      } else {
        next.add(categoryId);
      }
      return next;
    });
  };

  return (
    <div className="w-72 bg-gray-900 border-r border-gray-700 overflow-y-auto">
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-lg font-semibold text-white">Node Palette</h2>
        <p className="text-sm text-gray-400 mt-1">Drag nodes to canvas</p>
      </div>

      <div className="p-2">
        {nodeCategories.map((category) => {
          const Icon = category.icon;
          const isExpanded = expandedCategories.has(category.id);

          return (
            <div key={category.id} className="mb-2">
              <button
                onClick={() => toggleCategory(category.id)}
                className="w-full flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-800 transition-colors text-left"
              >
                {isExpanded ? (
                  <ChevronDown className="w-4 h-4 text-gray-400" />
                ) : (
                  <ChevronRight className="w-4 h-4 text-gray-400" />
                )}
                <Icon className="w-5 h-5 text-blue-400" />
                <span className="text-sm font-medium text-gray-200 flex-1">
                  {category.label}
                </span>
                <span className="text-xs text-gray-500 bg-gray-800 px-2 py-0.5 rounded">
                  {category.nodes.length}
                </span>
              </button>

              {isExpanded && (
                <div className="mt-1 ml-2 space-y-1">
                  {category.nodes.map((node) => (
                    <NodeCard
                      key={node.type}
                      node={node}
                      onDragStart={onNodeDragStart}
                    />
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

interface NodeCardProps {
  node: NodeDefinition;
  onDragStart: (event: React.DragEvent, nodeType: string) => void;
}

const NodeCard = ({ node, onDragStart }: NodeCardProps) => {
  const Icon = node.icon;

  return (
    <div
      draggable
      onDragStart={(e) => onDragStart(e, node.type)}
      className="group relative px-3 py-2.5 rounded-lg border border-gray-700 bg-gray-800/50 hover:bg-gray-800 hover:border-gray-600 transition-all cursor-grab active:cursor-grabbing"
      style={{
        borderLeftWidth: "3px",
        borderLeftColor: node.color,
      }}
    >
      <div className="flex items-start gap-2">
        <div
          className="p-1.5 rounded-md flex-shrink-0"
          style={{ backgroundColor: `${node.color}20` }}
        >
          <Icon className="w-4 h-4" style={{ color: node.color }} />
        </div>
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-medium text-gray-200 truncate">
            {node.label}
          </h4>
          <p className="text-xs text-gray-500 mt-0.5 line-clamp-2">
            {node.description}
          </p>
        </div>
      </div>
    </div>
  );
};
