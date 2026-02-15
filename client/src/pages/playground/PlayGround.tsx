import { useState, useEffect } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { useParams, useNavigate } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { getProjectById } from "../../lib/api/projectApi";
import toast, { Toaster } from "react-hot-toast";
import "@xyflow/react/dist/style.css";
import { Sidebar } from "../../components/playground/Sidebar";
import { Canvas } from "../../components/playground/Canvas";
import { ConfigModal } from "../../components/playground/ConfigModal";
import { ChatbotModal } from "../../components/playground/ChatbotModal";
import { ViewNodeModal } from "../../components/playground/ViewNodeModal";
import { ShareModal } from "../../components/playground/ShareModal";
import { Toolbar } from "../../components/playground/Toolbar";
import { ResultsPanel } from "../../components/playground/ResultsPanel";
import { usePlaygroundStore } from "../../store/playgroundStore";
import { executePipeline } from "../../features/playground/api";
import type { NodeType, BaseNodeData } from "../../types/pipeline";
import { useProjectState } from "../../hooks/queries/useProjectState";
import { useSaveProject } from "../../hooks/mutations/useSaveProject";
import { validatePipeline } from "../../utils/validation";
import { MentorAssistant, mentorApi } from "../../features/mentor";
import { useAuthStore } from "../../store/authStore";
import { useMentorContext } from "../../features/mentor";
import {
  useMentorStore,
  type MentorAction,
} from "../../features/mentor/store/mentorStore";

export default function PlayGround() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [viewNodeId, setViewNodeId] = useState<string | null>(null);
  const [chatbotNodeId, setChatbotNodeId] = useState<string | null>(null);
  const [resultsOpen, setResultsOpen] = useState(false);
  const [shareModalOpen, setShareModalOpen] = useState(false);

  const [executionProgress, setExecutionProgress] = useState<{
    status: string;
    percent: number;
    current_node?: number;
    total_nodes?: number;
  } | null>(null);

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
    updateNode,
    addNode,
  } = usePlaygroundStore();

  const { user } = useAuthStore();
  const { preferences } = useMentorStore();

  // Enable AI mentor context awareness
  useMentorContext({
    enabled: preferences.enabled,
    debounceMs: 1000,
  });

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
      // If configured/executed and has execution results, show view modal
      const nodeResult = executionResult?.nodeResults?.[nodeId];
      if (nodeResult && (node.data.isConfigured || nodeResult.success)) {
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

  const handleTemplateClick = async (templateId: string) => {
    const { getTemplateById } = await import("../../config/templateConfig");
    const { getNodeByType } = await import("../../config/nodeDefinitions");
    const template = getTemplateById(templateId);

    if (!template) {
      console.error("Template not found:", templateId);
      return;
    }

    // Generate unique IDs for nodes
    const nodeIdMap = new Map<number, string>();
    const newNodes = template.nodes
      .map((templateNode, index: number) => {
        const nodeId = `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        nodeIdMap.set(index, nodeId);

        const nodeDef = getNodeByType(templateNode.type);
        if (!nodeDef) {
          console.error(
            `Node definition not found for type: ${templateNode.type}`,
          );
          return null;
        }

        return {
          id: nodeId,
          type: templateNode.type, // Use the actual node type (e.g., 'upload_file')
          position: templateNode.position,
          data: {
            label: nodeDef.label,
            type: nodeDef.type,
            config: JSON.parse(JSON.stringify(nodeDef.defaultConfig)),
            isConfigured: false,
            color: nodeDef.color,
            icon: nodeDef.icon,
          },
        };
      })
      .filter(Boolean) as any[]; // Filter out nulls

    // Generate edges based on template connections
    const newEdges = template.edges.map((edge) => {
      const sourceId = nodeIdMap.get(edge.sourceIndex);
      const targetId = nodeIdMap.get(edge.targetIndex);

      return {
        id: `edge_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        source: sourceId!,
        target: targetId!,
        type: "smoothstep",
        animated: true,
      };
    });

    // Add nodes and edges to the canvas using the store
    const { setNodes, setEdges } = usePlaygroundStore.getState();
    setNodes((prevNodes) => [...prevNodes, ...newNodes]);
    setEdges((prevEdges) => [...prevEdges, ...newEdges]);

    toast.success(`${template.label} template loaded!`, {
      duration: 2000,
      position: "top-right",
      icon: "✅",
    });
  };

  const handleExecute = async () => {
    try {
      setIsExecuting(true);
      setResultsOpen(true);

      // 🛡️ VALIDATION: Check pipeline before execution
      const validationResult = validatePipeline(nodes, edges);

      if (!validationResult.isValid) {
        // Show validation errors as toast notifications
        validationResult.errors.forEach((error) => {
          const message = error.suggestion
            ? `${error.message}\n💡 ${error.suggestion}`
            : error.message;

          toast.error(message, {
            duration: 3000,
            position: "top-right",
            style: {
              background: "#1F2937",
              color: "#fff",
              border: "1px solid #EF4444",
              borderRadius: "8px",
              padding: "16px",
              maxWidth: "400px",
            },
          });
        });

        setIsExecuting(false);
        return;
      }

      // Show warnings if any
      const warnings = validationResult.errors.filter(
        (e) => e.type === "warning",
      );
      if (warnings.length > 0) {
        warnings.forEach((warning) => {
          const message = warning.suggestion
            ? `${warning.message}\n💡 ${warning.suggestion}`
            : warning.message;

          toast(message, {
            duration: 3000,
            position: "top-right",
            icon: "⚠️",
            style: {
              background: "#1F2937",
              color: "#fff",
              border: "1px solid #F59E0B",
              borderRadius: "8px",
              padding: "16px",
              maxWidth: "400px",
            },
          });
        });
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

      // Debug: Log each node's config
      pipelineConfig.forEach((nodeConfig, index) => {
        console.log(`📋 Node ${index + 1} (${nodeConfig.node_type}):`, {
          node_id: nodeConfig.node_id,
          input: nodeConfig.input,
        });
        if (nodeConfig.node_type === "transformation") {
          console.log("  ⚠️ Transformation columns:", nodeConfig.input.columns);
        }
        if (nodeConfig.node_type === "feature_selection") {
          console.log(
            "  🎯 Feature Selection - target_column:",
            nodeConfig.input.target_column,
          );
          console.log(
            "  🎯 Feature Selection - scoring_function:",
            nodeConfig.input.scoring_function,
          );
          console.log(
            "  🎯 Feature Selection - method:",
            nodeConfig.input.method,
          );
        }
      });

      const result = await executePipeline({
        pipeline: pipelineConfig,
        edges: edges.map((edge) => ({
          source: edge.source,
          target: edge.target,
          sourceHandle: edge.sourceHandle || undefined,
          targetHandle: edge.targetHandle || undefined,
        })),
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

      // Update each node with its execution result
      sortedNodes.forEach((node, index) => {
        const nodeResult = result.results?.[index];
        if (nodeResult && node.id) {
          updateNode(node.id, {
            result: nodeResult,
          });
        }
      });

      setExecutionResult({
        success: result.success,
        nodeResults,
        timestamp: new Date().toISOString(),
      });

      setExecutionProgress(null);
    } catch (error) {
      console.error("Pipeline execution failed:", error);

      setExecutionProgress(null);

      // Format user-friendly error message with details and suggestion
      let userFriendlyError = "Pipeline execution failed";
      let errorDetails: Record<string, unknown> | undefined = undefined;
      let errorSuggestion: string | undefined = undefined;

      if (error && typeof error === "object" && "response" in error) {
        const responseError = (error as any).response?.data;

        // FastAPI returns errors in "detail" field by default
        const errorMsg =
          responseError?.detail ||
          responseError?.message ||
          responseError?.error;

        if (errorMsg) {
          // Handle specific error types with user-friendly messages
          if (errorMsg.includes("Missing values found")) {
            const columnMatch = errorMsg.match(/column '([^']+)'/);
            const column = columnMatch ? columnMatch[1] : "a column";
            userFriendlyError = `Missing values detected in "${column}"`;
            errorSuggestion = "Please handle missing values before encoding.";
          } else if (errorMsg.includes("column_configs")) {
            userFriendlyError = "No columns configured for encoding";
            errorSuggestion =
              "Please configure encoding settings for at least one column.";
          } else if (
            errorMsg.includes("Dataset") &&
            errorMsg.includes("not found")
          ) {
            userFriendlyError = "Dataset not found or empty";
            errorSuggestion =
              "Please ensure the data source is properly connected.";
          } else {
            // Extract just the reason part if it exists
            const reasonMatch = errorMsg.match(/\[.*?\]: (.+)/);
            userFriendlyError = reasonMatch ? reasonMatch[1] : errorMsg;
          }

          // Extract details and suggestion from response if available
          if (responseError.details) {
            errorDetails = responseError.details;
          }
          if (responseError.suggestion && !errorSuggestion) {
            errorSuggestion = responseError.suggestion;
          }
        }
      } else if (error instanceof Error) {
        userFriendlyError = error.message;
      }

      setExecutionResult({
        success: false,
        error: userFriendlyError,
        errorDetails: errorDetails,
        errorSuggestion: errorSuggestion,
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

  const handleShare = () => {
    if (!projectId) {
      toast.error("No project selected. Please create a project first.");
      return;
    }

    // Auto-save before sharing
    const state = getProjectState();
    saveProject.mutate(
      { id: projectId, state },
      {
        onSuccess: () => {
          setShareModalOpen(true);
        },
        onError: () => {
          toast.error("Please save the project before sharing");
        },
      },
    );
  };

  // Handle adding a node from mentor suggestions
  const handleAddNodeFromMentor = async (nodeType: string) => {
    console.log("[handleAddNodeFromMentor] Adding node type:", nodeType);
    try {
      const { getNodeByType } = await import("../../config/nodeDefinitions");
      const nodeDef = getNodeByType(nodeType as NodeType);

      if (!nodeDef) {
        console.error(
          "[handleAddNodeFromMentor] Node definition not found for:",
          nodeType,
        );
        toast.error(`Node type "${nodeType}" not found`);
        return;
      }

      console.log(
        "[handleAddNodeFromMentor] Node definition found:",
        nodeDef.label,
      );

      // Create a new node with a unique ID
      const nodeId = `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      // Calculate position - center of viewport with slight offset for each new node
      const position = {
        x: 300 + nodes.length * 20, // Offset to avoid overlap
        y: 200 + nodes.length * 20,
      };

      const newNode = {
        id: nodeId,
        type: nodeDef.type,
        position,
        data: {
          label: nodeDef.label,
          type: nodeDef.type,
          config: JSON.parse(JSON.stringify(nodeDef.defaultConfig)),
          isConfigured: false,
          color: nodeDef.color,
          icon: nodeDef.icon,
        },
      };

      addNode(newNode);
      toast.success(`Added ${nodeDef.label} to canvas`, {
        icon: "✨",
        duration: 2000,
      });
    } catch (error) {
      console.error("Error adding node from mentor:", error);
      toast.error("Failed to add node");
    }
  };

  // Handle execute pipeline action from mentor
  const handleExecutePipeline = () => {
    handleExecute();
  };

  return (
    <div className="h-screen flex flex-col bg-gray-950">
      <Toaster />
      <Toolbar
        onExecute={handleExecute}
        onClear={handleClear}
        onSave={handleSave}
        onShare={projectId ? handleShare : undefined}
        isExecuting={usePlaygroundStore.getState().isExecuting}
        executionProgress={executionProgress}
        projectName={projectData?.name}
        onBack={() => navigate("/dashboard")}
      />

      <div className="flex-1 flex overflow-hidden">
        <ReactFlowProvider>
          <Sidebar
            onNodeDragStart={onNodeDragStart}
            onTemplateClick={handleTemplateClick}
          />
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

      <ShareModal
        isOpen={shareModalOpen}
        onClose={() => setShareModalOpen(false)}
        projectId={projectId ? parseInt(projectId) : 0}
        projectName={projectData?.name || "Untitled Project"}
      />

      <MentorAssistant
        userName={user?.fullName || user?.emailId || "there"}
        onAction={async (action: MentorAction) => {
          console.log("[Mentor] Action clicked:", action);

          // Handle mentor suggestion actions
          if (action.type === "show_guide") {
            const modelType = action.payload?.model_type as string;

            // Check if this is the initial guide request or dataset guidance
            if (action.payload?.action) {
              // This is a dataset guidance action
              try {
                const guidanceResponse = await mentorApi.getDatasetGuidance(
                  action.payload.action as string,
                  modelType,
                  action.payload.next_message as string,
                );

                if (
                  guidanceResponse.success &&
                  guidanceResponse.suggestions.length > 0
                ) {
                  useMentorStore
                    .getState()
                    .showSuggestion(guidanceResponse.suggestions[0]);
                }
              } catch (error) {
                console.error(
                  "[Mentor] Error fetching dataset guidance:",
                  error,
                );
                toast.error("Failed to load guidance");
              }
            } else {
              // Initial guide request - show model introduction
              if (modelType) {
                try {
                  const introResponse =
                    await mentorApi.getModelIntroduction(modelType);

                  if (
                    introResponse.success &&
                    introResponse.suggestions.length > 0
                  ) {
                    useMentorStore
                      .getState()
                      .showSuggestion(introResponse.suggestions[0]);
                    toast.success(
                      `Learning about ${modelType.replace(/_/g, " ")}`,
                    );
                  }
                } catch (error) {
                  console.error("[Mentor] Error fetching introduction:", error);
                  toast.error("Failed to load introduction");
                }
              }
            }
          } else if (action.type === "learn_more") {
            // Handle learn_more actions (dataset options that show more info)
            if (action.payload?.action && action.payload?.model_type) {
              try {
                const guidanceResponse = await mentorApi.getDatasetGuidance(
                  action.payload.action as string,
                  action.payload.model_type as string,
                  (action.payload.next_message as string) || "",
                );

                if (
                  guidanceResponse.success &&
                  guidanceResponse.suggestions.length > 0
                ) {
                  useMentorStore
                    .getState()
                    .showSuggestion(guidanceResponse.suggestions[0]);
                }
              } catch (error) {
                console.error("[Mentor] Error fetching guidance:", error);
                toast.error("Failed to load guidance");
              }
            } else if (action.payload?.url) {
              console.log("[Mentor] Opening learn more:", action.payload.url);
              window.open(action.payload.url as string, "_blank");
            } else {
              // "Get Suggestions" button — trigger pipeline analysis
              try {
                const pipelineData = {
                  nodes: nodes.map((n) => ({
                    id: n.id,
                    type: n.data.type,
                    config: n.data,
                    position: n.position,
                  })),
                  edges: edges.map((e) => ({
                    id: e.id,
                    source: e.source,
                    target: e.target,
                  })),
                };

                const analysis =
                  await mentorApi.analyzePipeline(pipelineData);

                if (analysis.suggestions.length > 0) {
                  analysis.suggestions.slice(0, 2).forEach((suggestion) => {
                    useMentorStore
                      .getState()
                      .showSuggestion(suggestion);
                  });
                } else {
                  toast("No suggestions right now — keep building!", {
                    icon: "💡",
                    duration: 2000,
                  });
                }
              } catch (error) {
                console.error(
                  "[Mentor] Error fetching suggestions:",
                  error,
                );
                toast.error("Failed to get suggestions");
              }
            }
          } else if (action.type === "add_node") {
            // Check for both node_type and model_type in payload
            const nodeType = (action.payload?.node_type ||
              action.payload?.model_type) as string;
            if (nodeType) {
              console.log("[Mentor] Adding node:", nodeType);
              handleAddNodeFromMentor(nodeType);
            } else {
              console.warn(
                "[Mentor] No node_type or model_type in payload:",
                action.payload,
              );
            }
          } else if (action.type === "execute") {
            console.log("[Mentor] Executing pipeline");
            handleExecutePipeline();
          } else {
            console.log("[Mentor] Unknown action type:", action.type);
          }
        }}
      />
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
