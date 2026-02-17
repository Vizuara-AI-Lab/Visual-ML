import { useState, useMemo } from "react";
import { X, Copy, Check, Download, Code, FileText } from "lucide-react";
import { usePlaygroundStore } from "../../store/playgroundStore";
import {
  exportToPython,
  exportToNotebook,
  downloadFile,
  copyToClipboard,
} from "../../lib/export";
import type { ExportFormat } from "../../lib/export";

interface ExportModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ExportModal = ({ isOpen, onClose }: ExportModalProps) => {
  const [format, setFormat] = useState<ExportFormat>("python");
  const [isCopied, setIsCopied] = useState(false);

  const { nodes, edges } = usePlaygroundStore();

  // Generate code whenever nodes/edges/format change
  const generated = useMemo(() => {
    if (!isOpen || nodes.length === 0) return null;

    try {
      if (format === "python") {
        const pipeline = exportToPython(nodes, edges);
        return { content: pipeline.pythonSource, error: null };
      } else {
        const notebook = exportToNotebook(nodes, edges);
        return { content: notebook, error: null };
      }
    } catch (err) {
      return {
        content: "",
        error:
          err instanceof Error ? err.message : "Failed to generate code",
      };
    }
  }, [isOpen, nodes, edges, format]);

  const handleCopy = async () => {
    if (!generated?.content) return;

    const success = await copyToClipboard(generated.content);
    if (success) {
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    }
  };

  const handleDownload = () => {
    if (!generated?.content) return;

    if (format === "python") {
      downloadFile("pipeline.py", generated.content, "text/x-python");
    } else {
      downloadFile(
        "pipeline.ipynb",
        generated.content,
        "application/x-ipynb+json",
      );
    }
  };

  if (!isOpen) return null;

  const hasNodes = nodes.length > 0;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-3xl mx-4 overflow-hidden max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between bg-gradient-to-r from-emerald-50 to-white shrink-0">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-emerald-600 rounded-lg">
              <Code className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-slate-800">
                Export to Code
              </h2>
              <p className="text-sm text-slate-500">
                Generate executable Python code from your pipeline
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-slate-500" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-4 overflow-auto flex-1">
          {!hasNodes ? (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Code className="w-8 h-8 text-slate-400" />
              </div>
              <h3 className="text-lg font-semibold text-slate-800 mb-2">
                No Nodes on Canvas
              </h3>
              <p className="text-sm text-slate-500">
                Add some nodes to your pipeline and connect them to generate
                code.
              </p>
            </div>
          ) : (
            <>
              {/* Format Selection */}
              <div className="flex gap-3">
                <button
                  onClick={() => setFormat("python")}
                  className={`flex-1 flex items-center gap-3 p-4 rounded-xl border-2 transition-all ${
                    format === "python"
                      ? "border-emerald-500 bg-emerald-50"
                      : "border-slate-200 hover:border-slate-300 bg-white"
                  }`}
                >
                  <FileText
                    className={`w-5 h-5 ${format === "python" ? "text-emerald-600" : "text-slate-400"}`}
                  />
                  <div className="text-left">
                    <p
                      className={`font-semibold ${format === "python" ? "text-emerald-800" : "text-slate-700"}`}
                    >
                      Python Script
                    </p>
                    <p className="text-xs text-slate-500">.py file</p>
                  </div>
                </button>

                <button
                  onClick={() => setFormat("notebook")}
                  className={`flex-1 flex items-center gap-3 p-4 rounded-xl border-2 transition-all ${
                    format === "notebook"
                      ? "border-emerald-500 bg-emerald-50"
                      : "border-slate-200 hover:border-slate-300 bg-white"
                  }`}
                >
                  <FileText
                    className={`w-5 h-5 ${format === "notebook" ? "text-emerald-600" : "text-slate-400"}`}
                  />
                  <div className="text-left">
                    <p
                      className={`font-semibold ${format === "notebook" ? "text-emerald-800" : "text-slate-700"}`}
                    >
                      Jupyter Notebook
                    </p>
                    <p className="text-xs text-slate-500">.ipynb file</p>
                  </div>
                </button>
              </div>

              {/* Error */}
              {generated?.error && (
                <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                  {generated.error}
                </div>
              )}

              {/* Code Preview */}
              {generated?.content && (
                <div className="relative">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">
                      {format === "python" ? "Python" : "Notebook JSON"} Preview
                    </span>
                    <span className="text-xs text-slate-400">
                      {generated.content.split("\n").length} lines
                    </span>
                  </div>
                  <pre className="bg-slate-900 text-slate-100 rounded-xl p-4 text-sm font-mono overflow-auto max-h-[400px] leading-relaxed">
                    <code>
                      {format === "python"
                        ? generated.content
                        : // Show formatted Python for notebook preview too
                          (() => {
                            try {
                              const pipeline = exportToPython(nodes, edges);
                              return pipeline.pythonSource;
                            } catch {
                              return generated.content;
                            }
                          })()}
                    </code>
                  </pre>
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-slate-50 border-t border-slate-200 flex items-center justify-between shrink-0">
          <button
            onClick={onClose}
            className="px-4 py-2 text-slate-600 hover:text-slate-800 font-medium transition-colors"
          >
            Close
          </button>

          {hasNodes && generated?.content && (
            <div className="flex items-center gap-3">
              <button
                onClick={handleCopy}
                className="px-4 py-2 border border-slate-300 hover:border-slate-400 text-slate-700 rounded-lg flex items-center gap-2 transition-colors font-medium"
              >
                {isCopied ? (
                  <>
                    <Check className="w-4 h-4 text-emerald-600" />
                    <span className="text-emerald-600">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-4 h-4" />
                    Copy
                  </>
                )}
              </button>

              <button
                onClick={handleDownload}
                className="px-6 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg font-semibold transition-all shadow-lg hover:shadow-xl flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                Download {format === "python" ? ".py" : ".ipynb"}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
