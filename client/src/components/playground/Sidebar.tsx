import { useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  PanelLeftClose,
  PanelLeft,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import {
  nodeCategories,
  type NodeDefinition,
} from "../../config/nodeDefinitions";

interface SidebarProps {
  onNodeDragStart: (event: React.DragEvent, nodeType: string) => void;
}

export const Sidebar = ({ onNodeDragStart }: SidebarProps) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
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
    <motion.div
      initial={{ width: 288 }}
      animate={{ width: isCollapsed ? 48 : 288 }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      className="bg-white/90 backdrop-blur-xl border-r border-slate-200/60 overflow-y-auto shrink-0 shadow-lg shadow-slate-900/5"
    >
      {/* Collapsed State - Just Toggle Button */}
      {isCollapsed ? (
        <div className="h-full flex flex-col items-center py-4">
          <button
            onClick={() => setIsCollapsed(false)}
            className="p-2 rounded-lg hover:bg-slate-100/50 transition-colors group"
            title="Expand Sidebar"
          >
            <PanelLeft className="w-5 h-5 text-slate-500 group-hover:text-slate-800 transition-colors" />
          </button>
        </div>
      ) : (
        <>
          {/* Expanded State - Full Content */}
          <div className="p-4 border-b border-slate-200/60 flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-slate-800">
                Node Palette
              </h2>
              <p className="text-sm text-slate-500 mt-1">
                Drag nodes to canvas
              </p>
            </div>
            <button
              onClick={() => setIsCollapsed(true)}
              className="p-2 rounded-lg hover:bg-slate-100/50 transition-colors group"
              title="Collapse Sidebar"
            >
              <PanelLeftClose className="w-5 h-5 text-slate-500 group-hover:text-slate-800 transition-colors" />
            </button>
          </div>

          <div className="p-3">
            {nodeCategories.map((category) => {
              const Icon = category.icon;
              const isExpanded = expandedCategories.has(category.id);

              return (
                <div key={category.id} className="mb-2">
                  <button
                    onClick={() => toggleCategory(category.id)}
                    className="w-full flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-slate-100/50 transition-colors text-left"
                  >
                    {isExpanded ? (
                      <ChevronDown className="w-4 h-4 text-slate-500" />
                    ) : (
                      <ChevronRight className="w-4 h-4 text-slate-500" />
                    )}
                    <Icon className="w-5 h-5 text-slate-700" />
                    <span className="text-sm font-medium text-slate-800 flex-1">
                      {category.label}
                    </span>
                    <span className="text-xs text-slate-500 bg-slate-50 px-2 py-0.5 rounded">
                      {category.nodes.length}
                    </span>
                  </button>

                  <AnimatePresence>
                    {isExpanded && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="overflow-hidden"
                      >
                        <div className="mt-1 ml-2 space-y-1">
                          {category.nodes.map((node) => (
                            <NodeCard
                              key={node.type}
                              node={node}
                              onDragStart={onNodeDragStart}
                            />
                          ))}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              );
            })}
          </div>
        </>
      )}
    </motion.div>
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
      className="group relative px-3 py-2.5 rounded-lg border border-slate-200/60 bg-white/80 hover:bg-white hover:border-slate-300 transition-all cursor-grab active:cursor-grabbing shadow-sm hover:shadow-md"
      style={{
        borderLeftWidth: "3px",
        borderLeftColor: node.color,
      }}
    >
      <div className="flex items-start gap-2">
        <div
          className="p-1.5 rounded-md shrink-0"
          style={{ backgroundColor: `${node.color}20` }}
        >
          <Icon className="w-4 h-4" style={{ color: node.color }} />
        </div>
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-medium text-slate-800 truncate">
            {node.label}
          </h4>
          <p className="text-xs text-slate-500 mt-0.5 line-clamp-2">
            {node.description}
          </p>
        </div>
      </div>
    </div>
  );
};
