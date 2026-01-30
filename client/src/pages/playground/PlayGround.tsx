import { useState, useEffect } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { useParams, useNavigate } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { getProjectById } from "../../lib/api/projectApi";
import "@xyflow/react/dist/style.css";
import { Sidebar } from "../../components/playground/Sidebar";
import { Canvas } from "../../components/playground/Canvas";
import { ConfigModal } from "../../components/playground/ConfigModal";
import { ChatbotModal } from "../../components/playground/ChatbotModal";
import { ViewNodeModal } from "../../components/playground/ViewNodeModal";
import { Toolbar } from "../../components/playground/Toolbar";
import { ResultsPanel } from "../../components/playground/ResultsPanel";
import { ValidationDialog } from "../../components/playground/ValidationDialog";
import { usePlaygroundStore } from "../../store/playgroundStore";
import { executePipeline } from "../../features/playground/api";
import type { NodeType, BaseNodeData } from "../../types/pipeline";
import { useProjectState } from "../../hooks/queries/useProjectState";
import { useSaveProject } from "../../hooks/mutations/useSaveProject";
import { validatePipeline, type ValidationError } from "../../utils/validation";

export default function PlayGround() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [viewNodeId, setViewNodeId] = useState<string | null>(null);
  const [chatbotNodeId, setChatbotNodeId] = useState<string | null>(null);
  const [resultsOpen, setResultsOpen] = useState(false);
  const [validationErrors, setValidationErrors] = useState<ValidationError[]>(
    [],
  );

  const {
    nodes,
    edges,
    setExecutionResult,
    setIsExecuting,
    clearAll,
    setCurrentProjectId,
    loadProjectState,
    getProjectState,
    executionResult,
    currentProjectId,
  } = usePlaygroundStore();

  // Handle node click - show view data modal for view nodes, config for others
  const handleNodeClick = (nodeId: string) => {
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return;

    const viewNodeTypes = [
      "table_view",
      "data_preview",
      "statistics_view",
      "column_info",
      "chart_view",
    ];

    // If it's a chatbot node, show chat interface
    if (node.data.type === "chatbot_node") {
      setChatbotNodeId(nodeId);
      return;
    }

    // If it's a view node
    if (viewNodeTypes.includes(node.data.type)) {
      // If configured and has execution results, show view modal
      if (node.data.isConfigured && executionResult?.nodeResults?.[nodeId]) {
        setViewNodeId(nodeId);
      } else {
        // Otherwise show config modal
        setSelectedNodeId(nodeId);
      }
    } else {
      // For non-view nodes, always show config modal
      setSelectedNodeId(nodeId);
    }
  };

  // Load project state if projectId exists
  const { data: projectStateData } = useProjectState(projectId);
  const { data: projectData } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => getProjectById(projectId!),
    enabled: !!projectId,
  });
  const saveProject = useSaveProject();

  // Set current project ID and load state
  useEffect(() => {
    if (projectId) {
      setCurrentProjectId(projectId);
    }
  }, [projectId, setCurrentProjectId]);

  // Load project state when data is available
  useEffect(() => {
    if (projectStateData?.state) {
      loadProjectState(projectStateData.state);
    }
  }, [projectStateData, loadProjectState]);

  // Listen for custom event from View Data button on nodes
  useEffect(() => {
    const handleOpenViewModal = (event: CustomEvent) => {
      const { nodeId } = event.detail;
      setViewNodeId(nodeId);
    };

    window.addEventListener(
      "openViewNodeModal",
      handleOpenViewModal as EventListener,
    );

    return () => {
      window.removeEventListener(
        "openViewNodeModal",
        handleOpenViewModal as EventListener,
      );
    };
  }, []);

  // Listen for custom event from Reconfig button on nodes
  useEffect(() => {
    const handleOpenConfigModal = (event: CustomEvent) => {
      const { nodeId } = event.detail;
      setSelectedNodeId(nodeId);
    };

    window.addEventListener(
      "openConfigModal",
      handleOpenConfigModal as EventListener,
    );

    return () => {
      window.removeEventListener(
        "openConfigModal",
        handleOpenConfigModal as EventListener,
      );
    };
  }, []);

  const onNodeDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData("application/reactflow", nodeType);
    event.dataTransfer.effectAllowed = "move";
  };

  const handleExecute = async () => {
    try {
      setIsExecuting(true);
      setResultsOpen(true);

      // 🛡️ VALIDATION: Check pipeline before execution
      const validationResult = validatePipeline(nodes, edges);

      if (!validationResult.isValid) {
        setValidationErrors(validationResult.errors);

        setExecutionResult({
          success: false,
          error:
            "Pipeline validation failed. Please fix the errors and try again.",
          nodeResults: {},
          timestamp: new Date().toISOString(),
        });

        setIsExecuting(false);
        return;
      }

      // Show warnings if any
      const warnings = validationResult.errors.filter(
        (e) => e.type === "warning",
      );
      if (warnings.length > 0) {
        setValidationErrors(warnings);
        setIsExecuting(false);
        return;
      }

      // Topological sort to determine execution order
      const sortedNodes = topologicalSort(nodes, edges);

      // Inject project_id into each node's input for nodes that need it (upload_file)
      const pipelineConfig = sortedNodes.map((node) => ({
        node_id: node.id,
        node_type: node.type as NodeType,
        input: {
          ...(node.data.config || {}),
          // Add project_id if available for nodes that save datasets
          ...(currentProjectId && ["upload_file"].includes(node.type as string)
            ? { project_id: parseInt(currentProjectId) }
            : {}),
        },
      }));

      console.log("🚀 Executing pipeline with config:", pipelineConfig);

      const result = await executePipeline({
        pipeline: pipelineConfig,
        pipeline_name: "Playground Pipeline",
      });

      console.log("✅ Pipeline execution result:", result);

      // Store results by node ID
      const nodeResults: Record<string, any> = {};
      if (result.results) {
        result.results.forEach((nodeResult: any, index: number) => {
          const nodeId = sortedNodes[index]?.id;
          if (nodeId) {
            console.log(
              `📦 Node ${nodeId} (${sortedNodes[index]?.type}) result:`,
              nodeResult,
            );
            nodeResults[nodeId] = nodeResult;
          }
        });
      }

      console.log("📊 All node results:", nodeResults);

      setExecutionResult({
        success: result.success,
        nodeResults,
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      console.error("Pipeline execution failed:", error);
      setExecutionResult({
        success: false,
        error:
          error instanceof Error ? error.message : "Unknown error occurred",
        timestamp: new Date().toISOString(),
      });
    } finally {
      setIsExecuting(false);
    }
  };

  const handleClear = () => {
    if (window.confirm("Are you sure you want to clear the entire canvas?")) {
      clearAll();
      setSelectedNodeId(null);
      setResultsOpen(false);
    }
  };

  const handleSave = () => {
    if (!projectId) {
      alert("No project selected. Please create a project first.");
      return;
    }

    const state = getProjectState();
    saveProject.mutate(
      { id: projectId, state },
      {
        onSuccess: () => {
          console.log("✅ Project saved successfully");
        },
      },
    );
  };

  const handleLoad = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const data = JSON.parse(event.target?.result as string);
          // TODO: Load nodes and edges into store
          console.log("Loaded pipeline:", data);
        } catch (error) {
          console.error("Failed to load pipeline:", error);
        }
      };
      reader.readAsText(file);
    };
    input.click();
  };

  const handleExport = () => {
    // Export as Python code or other format
    console.log("Export functionality coming soon");
  };

  return (
    <div className="h-screen flex flex-col bg-gray-950">
      <Toolbar
        onExecute={handleExecute}
        onClear={handleClear}
        onSave={handleSave}
        onLoad={handleLoad}
        onExport={handleExport}
        isExecuting={usePlaygroundStore.getState().isExecuting}
        projectName={projectData?.name}
        onBack={() => navigate("/dashboard")}
      />

      <div className="flex-1 flex overflow-hidden">
        <ReactFlowProvider>
          <Sidebar onNodeDragStart={onNodeDragStart} />
          <Canvas onNodeClick={handleNodeClick} />
        </ReactFlowProvider>

        <ResultsPanel
          isOpen={resultsOpen}
          onClose={() => setResultsOpen(false)}
        />
      </div>

      <ConfigModal
        nodeId={selectedNodeId}
        onClose={() => setSelectedNodeId(null)}
      />

      <ViewNodeModal nodeId={viewNodeId} onClose={() => setViewNodeId(null)} />

      <ChatbotModal
        nodeId={chatbotNodeId}
        onClose={() => setChatbotNodeId(null)}
      />

      {validationErrors.length > 0 && (
        <ValidationDialog
          errors={validationErrors}
          onClose={() => {
            setValidationErrors([]);
            setIsExecuting(false);
          }}
          onProceed={
            validationErrors.every((e) => e.type === "warning")
              ? async () => {
                  setValidationErrors([]);
                  // Continue with execution
                  try {
                    const pipelineNodes = nodes.map((node) => ({
                      id: node.id,
                      type: node.type as NodeType,
                      config: node.data.config || {},
                    }));

                    const pipelineEdges = edges.map((edge) => ({
                      source: edge.source,
                      target: edge.target,
                    }));

                    const result = await executePipeline(
                      pipelineNodes,
                      pipelineEdges,
                    );

                    setExecutionResult({
                      success: true,
                      nodeResults: result.results,
                      timestamp: new Date().toISOString(),
                    });

                    console.log("✅ Execution completed:", result);
                  } catch (error: any) {
                    console.error("❌ Execution failed:", error);
                    setExecutionResult({
                      success: false,
                      error: error.message || "Unknown error occurred",
                      nodeResults: {},
                      timestamp: new Date().toISOString(),
                    });
                  } finally {
                    setIsExecuting(false);
                  }
                }
              : undefined
          }
        />
      )}
    </div>
  );
}

// Topological sort helper
function topologicalSort(
  nodes: Array<{
    id: string;
    type?: string;
    data: BaseNodeData;
  }>,
  edges: Array<{ source: string; target: string }>,
): Array<{
  id: string;
  type?: string;
  data: BaseNodeData;
}> {
  const adjList = new Map<string, string[]>();
  const inDegree = new Map<string, number>();

  nodes.forEach((node) => {
    adjList.set(node.id, []);
    inDegree.set(node.id, 0);
  });

  edges.forEach((edge) => {
    adjList.get(edge.source)?.push(edge.target);
    inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1);
  });

  const queue: string[] = [];
  inDegree.forEach((degree, nodeId) => {
    if (degree === 0) queue.push(nodeId);
  });

  const sorted: typeof nodes = [];
  while (queue.length > 0) {
    const nodeId = queue.shift()!;
    const node = nodes.find((n) => n.id === nodeId);
    if (node) sorted.push(node);

    adjList.get(nodeId)?.forEach((neighbor) => {
      const newDegree = (inDegree.get(neighbor) || 0) - 1;
      inDegree.set(neighbor, newDegree);
      if (newDegree === 0) queue.push(neighbor);
    });
  }

  return sorted;
}
