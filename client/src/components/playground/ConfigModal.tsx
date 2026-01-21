import { motion, AnimatePresence } from "framer-motion";
import { X, Save } from "lucide-react";
import { useState } from "react";
import { getNodeByType } from "../../config/nodeDefinitions";
import { usePlaygroundStore } from "../../store/playgroundStore";

interface ConfigModalProps {
  nodeId: string | null;
  onClose: () => void;
}

export const ConfigModal = ({ nodeId, onClose }: ConfigModalProps) => {
  const { getNodeById, updateNodeConfig, datasetMetadata } =
    usePlaygroundStore();
  const node = nodeId ? getNodeById(nodeId) : null;
  const nodeDef = node ? getNodeByType(node.type) : null;

  const [config, setConfig] = useState<Record<string, unknown>>(
    () => node?.data.config || {},
  );

  if (!node || !nodeDef) return null;

  const handleSave = () => {
    updateNodeConfig(node.id, config);
    onClose();
  };

  const handleFieldChange = (fieldName: string, value: unknown) => {
    setConfig((prev) => ({ ...prev, [fieldName]: value }));
  };

  // Auto-fill logic based on dataset metadata
  const getAutoFilledValue = (fieldName: string): unknown => {
    if (!datasetMetadata) return undefined;

    if (fieldName === "dataset_path" && datasetMetadata.dataset_id) {
      return datasetMetadata.dataset_id;
    }
    if (fieldName === "target_column" && datasetMetadata.suggested_target) {
      return datasetMetadata.suggested_target;
    }
    return undefined;
  };

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 bg-black/60 backdrop-blur-sm"
          onClick={onClose}
        />

        <motion.div
          initial={{ scale: 0.95, opacity: 0, y: 20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.95, opacity: 0, y: 20 }}
          transition={{ type: "spring", damping: 25, stiffness: 300 }}
          className="relative bg-gray-900 border border-gray-700 rounded-xl shadow-2xl max-w-2xl w-full max-h-[80vh] overflow-hidden z-10"
        >
          {/* Header */}
          <div
            className="px-6 py-4 border-b border-gray-700 flex items-center justify-between"
            style={{ borderTopColor: nodeDef.color, borderTopWidth: "3px" }}
          >
            <div className="flex items-center gap-3">
              <div
                className="p-2 rounded-lg"
                style={{ backgroundColor: `${nodeDef.color}20` }}
              >
                <nodeDef.icon
                  className="w-5 h-5"
                  style={{ color: nodeDef.color }}
                />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">
                  {nodeDef.label}
                </h3>
                <p className="text-sm text-gray-400">{nodeDef.description}</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-gray-400" />
            </button>
          </div>

          {/* Body */}
          <div className="p-6 overflow-y-auto max-h-[calc(80vh-180px)]">
            {nodeDef.configFields && nodeDef.configFields.length > 0 ? (
              <div className="space-y-4">
                {nodeDef.configFields.map((field) => {
                  const autoFilledValue = field.autoFill
                    ? getAutoFilledValue(field.name)
                    : undefined;
                  const currentValue =
                    config[field.name] ?? autoFilledValue ?? field.defaultValue;

                  return (
                    <div key={field.name}>
                      <label className="block text-sm font-medium text-gray-200 mb-2">
                        {field.label}
                        {field.required && (
                          <span className="text-red-400 ml-1">*</span>
                        )}
                      </label>

                      {field.type === "text" && (
                        <input
                          type="text"
                          value={(currentValue as string) || ""}
                          onChange={(e) =>
                            handleFieldChange(field.name, e.target.value)
                          }
                          className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                          placeholder={field.description}
                        />
                      )}

                      {field.type === "number" && (
                        <input
                          type="number"
                          value={(currentValue as number) || 0}
                          onChange={(e) =>
                            handleFieldChange(
                              field.name,
                              parseFloat(e.target.value),
                            )
                          }
                          min={field.min}
                          max={field.max}
                          step={field.step}
                          className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      )}

                      {field.type === "select" && (
                        <select
                          value={(currentValue as string) || ""}
                          onChange={(e) =>
                            handleFieldChange(field.name, e.target.value)
                          }
                          className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                          <option value="">Select {field.label}</option>
                          {field.options?.map((opt) => (
                            <option key={opt.value} value={opt.value}>
                              {opt.label}
                            </option>
                          ))}
                          {field.autoFill &&
                            datasetMetadata?.columns &&
                            field.name === "target_column" &&
                            datasetMetadata.columns.map((col: string) => (
                              <option key={col} value={col}>
                                {col}
                              </option>
                            ))}
                        </select>
                      )}

                      {field.type === "textarea" && (
                        <textarea
                          value={(currentValue as string) || ""}
                          onChange={(e) =>
                            handleFieldChange(field.name, e.target.value)
                          }
                          rows={4}
                          className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                          placeholder={field.description}
                        />
                      )}

                      {field.type === "checkbox" && (
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={(currentValue as boolean) || false}
                            onChange={(e) =>
                              handleFieldChange(field.name, e.target.checked)
                            }
                            className="w-4 h-4 bg-gray-800 border-gray-600 rounded text-blue-500 focus:ring-2 focus:ring-blue-500"
                          />
                          <span className="text-sm text-gray-400">
                            {field.description}
                          </span>
                        </label>
                      )}

                      {field.type === "file" && (
                        <input
                          type="file"
                          onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (file) {
                              handleFieldChange(field.name, file);
                            }
                          }}
                          className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-blue-600 file:text-white file:cursor-pointer hover:file:bg-blue-700"
                        />
                      )}

                      {field.description && field.type !== "checkbox" && (
                        <p className="text-xs text-gray-500 mt-1">
                          {field.description}
                        </p>
                      )}

                      {autoFilledValue && (
                        <p className="text-xs text-green-400 mt-1">
                          Auto-filled from dataset
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>No configuration required for this node</p>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="px-6 py-4 border-t border-gray-700 flex items-center justify-end gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg flex items-center gap-2 transition-colors"
            >
              <Save className="w-4 h-4" />
              Save Configuration
            </button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};
