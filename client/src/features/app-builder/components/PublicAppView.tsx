/**
 * PublicAppView — Renders a published custom app for end users.
 * No authentication. Collects input, executes pipeline, shows results.
 */

import { useState } from "react";
import { toast } from "react-hot-toast";
import { useExecutePublicApp } from "../hooks/usePublicApp";
import BlockRenderer from "./BlockRenderer";
import type { PublicApp, AppTheme, ExecuteAppResponse } from "../types/appBuilder";

interface PublicAppViewProps {
  app: PublicApp;
  slug: string;
}

export default function PublicAppView({ app, slug }: PublicAppViewProps) {
  const executeMutation = useExecutePublicApp(slug);
  const [formData, setFormData] = useState<Record<string, unknown>>({});
  const [fileData, setFileData] = useState<string | null>(null);
  const [fileNodeId, setFileNodeId] = useState<string | null>(null);
  const [nodeInputs, setNodeInputs] = useState<Record<string, Record<string, unknown>>>({});
  const [results, setResults] = useState<ExecuteAppResponse | null>(null);

  const handleFieldChange = (name: string, value: unknown, nodeId?: string, nodeConfigKey?: string) => {
    setFormData((prev) => ({ ...prev, [name]: value }));
    if (nodeId && nodeConfigKey) {
      setNodeInputs((prev) => ({
        ...prev,
        [nodeId]: { ...(prev[nodeId] || {}), [nodeConfigKey]: value },
      }));
    }
  };

  const handleFileUpload = (base64: string, filename: string, nodeId?: string) => {
    setFileData(base64);
    setFormData((prev) => ({ ...prev, filename }));
    if (nodeId) setFileNodeId(nodeId);
  };

  const handleSubmit = async () => {
    try {
      const hasNodeInputs = Object.keys(nodeInputs).length > 0;
      const res = await executeMutation.mutateAsync({
        input_data: formData,
        file_data: fileData ?? undefined,
        node_inputs: hasNodeInputs ? nodeInputs : undefined,
        file_node_id: fileNodeId ?? undefined,
      });
      setResults(res);
      if (res.success) {
        toast.success("Pipeline executed successfully");
      } else {
        toast.error(res.error ?? "Execution failed");
      }
    } catch {
      toast.error("Failed to execute pipeline");
    }
  };

  const theme = app.theme ?? { primaryColor: "#6366f1", fontFamily: "Inter", darkMode: false };

  return (
    <div
      className={`min-h-screen ${theme.darkMode ? "bg-gray-900 text-white" : "bg-gray-50 text-gray-900"}`}
      style={{ fontFamily: theme.fontFamily }}
    >
      <div className="max-w-3xl mx-auto px-4 py-8 space-y-6">
        {app.blocks.map((block) => (
          <BlockRenderer
            key={block.id}
            block={block}
            mode="live"
            theme={theme}
            formData={formData}
            results={results}
            isExecuting={executeMutation.isPending}
            onFieldChange={handleFieldChange}
            onFileUpload={handleFileUpload}
            onSubmit={handleSubmit}
          />
        ))}
      </div>

      {/* Footer */}
      <div className="text-center py-6 border-t mt-12">
        <p className={`text-xs ${theme.darkMode ? "text-gray-500" : "text-gray-400"}`}>
          Built by {app.owner_name} &middot; Powered by Visual-ML
        </p>
      </div>
    </div>
  );
}
