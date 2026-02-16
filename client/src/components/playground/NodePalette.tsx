/**
 * Node Palette - Draggable nodes for the pipeline builder
 */

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, Sparkles } from "lucide-react";
import { getAllNodes } from "../../config/nodeDefinitions";

const NodePalette = () => {
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(["data-sources", "preprocessing", "ml-algorithms"]),
  );

  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData("application/reactflow", nodeType);
    event.dataTransfer.effectAllowed = "move";
  };

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

  // Get all nodes and organize by category
  const allNodes = getAllNodes();

  const categories = [
    { id: "data-sources", label: "Data Sources", icon: "📁" },
    { id: "view", label: "View Data", icon: "👁️" },
    { id: "preprocessing", label: "Preprocess", icon: "🧹" },
    { id: "feature-engineering", label: "Feature Engineering", icon: "⚙️" },
    { id: "target-split", label: "Target & Split", icon: "🎯" },
    { id: "ml-algorithms", label: "Model Training", icon: "🤖" },
    { id: "result", label: "Results & Metrics", icon: "📊" },
    { id: "genai", label: "GenAI", icon: "✨" },
    { id: "deployment", label: "Deployment", icon: "🚀" },
  ];

  return (
    <div className="w-72 bg-linear-to-b from-slate-50/95 to-white/95 backdrop-blur-xl border-r border-slate-200/60 overflow-y-auto shadow-xl">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-linear-to-br from-slate-900 to-slate-700 px-4 py-4 shadow-lg shadow-slate-900/25">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-white/10 backdrop-blur-sm border border-white/20 flex items-center justify-center shadow-lg">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">Node Palette</h2>
            <p className="text-xs text-slate-300">
              Drag nodes to build your pipeline
            </p>
          </div>
        </div>
      </div>

      {/* Categories */}
      <div className="p-4 space-y-3">
        {categories.map((category) => {
          const nodes = allNodes.filter(
            (node) => node.category === category.id,
          );
          if (nodes.length === 0) return null;

          const isExpanded = expandedCategories.has(category.id);

          return (
            <div key={category.id} className="space-y-2">
              {/* Category Header */}
              <button
                onClick={() => toggleCategory(category.id)}
                className="w-full flex items-center justify-between px-3 py-2.5 rounded-lg bg-white/80 hover:bg-white border border-slate-200/60 hover:border-slate-300 transition-all group shadow-sm hover:shadow-md"
              >
                <div className="flex items-center gap-2">
                  <span className="text-base">{category.icon}</span>
                  <span className="text-sm font-semibold text-slate-800 group-hover:text-slate-900">
                    {category.label}
                  </span>
                  <span className="ml-1 px-1.5 py-0.5 bg-slate-100 text-slate-600 rounded text-xs font-medium">
                    {nodes.length}
                  </span>
                </div>
                <motion.div
                  animate={{ rotate: isExpanded ? 0 : -90 }}
                  transition={{ duration: 0.2 }}
                >
                  <ChevronDown className="w-4 h-4 text-slate-500 group-hover:text-slate-700" />
                </motion.div>
              </button>

              {/* Category Nodes */}
              <AnimatePresence>
                {isExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="space-y-2 pl-2">
                      {nodes.map((node, index) => (
                        <motion.div
                          key={node.type}
                          initial={{ x: -10, opacity: 0 }}
                          animate={{ x: 0, opacity: 1 }}
                          transition={{ duration: 0.2, delay: index * 0.05 }}
                          className="group"
                        >
                          <div
                            draggable
                            onDragStart={(e) => onDragStart(e, node.type)}
                            className="relative p-3 bg-white/70 backdrop-blur-sm rounded-lg border border-slate-200/60 cursor-move hover:bg-white hover:border-slate-300 hover:shadow-lg transition-all overflow-hidden"
                          >
                            {/* Accent bar */}
                            <div
                              className="absolute left-0 top-0 bottom-0 w-1 group-hover:w-1.5 transition-all"
                              style={{ backgroundColor: node.color }}
                            />

                            {/* Gradient overlay on hover */}
                            <div className="absolute inset-0 bg-linear-to-br from-slate-50/50 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />

                            <div className="relative flex items-start gap-3">
                              {/* Icon */}
                              <div
                                className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0 shadow-sm group-hover:shadow-md transition-shadow"
                                style={{
                                  backgroundColor: `${node.color}15`,
                                }}
                              >
                                {typeof node.icon === "string" ? (
                                  <span className="text-lg">{node.icon}</span>
                                ) : (
                                  <node.icon
                                    className="w-5 h-5"
                                    style={{ color: node.color }}
                                  />
                                )}
                              </div>

                              {/* Content */}
                              <div className="flex-1 min-w-0">
                                <div className="text-sm font-semibold text-slate-800 group-hover:text-slate-900 transition-colors">
                                  {node.label}
                                </div>
                                <div className="text-xs text-slate-600 mt-0.5 line-clamp-2 leading-relaxed">
                                  {node.description}
                                </div>
                              </div>
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>

      {/* Bottom Tip */}
      <div className="sticky bottom-0 p-4 bg-linear-to-t from-white via-white to-transparent">
        <div className="p-3 bg-linear-to-br from-slate-900 to-slate-700 rounded-xl border border-slate-800 shadow-lg shadow-slate-900/25">
          <div className="flex items-start gap-2">
            <Sparkles className="w-4 h-4 text-slate-300 shrink-0 mt-0.5" />
            <p className="text-xs text-slate-200 leading-relaxed">
              <strong className="text-white font-semibold">Quick Tip:</strong>{" "}
              Drag and drop nodes onto the canvas to start building your ML
              pipeline
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NodePalette;
