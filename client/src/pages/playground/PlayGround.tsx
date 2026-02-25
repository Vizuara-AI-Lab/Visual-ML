import { useState, useEffect, useRef } from "react";
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
import { ExportModal } from "../../components/playground/ExportModal";
import { ActivityModal } from "../../components/playground/activities/ActivityModal";
import { Toolbar } from "../../components/playground/Toolbar";
import { ResultsDrawer } from "../../components/playground/results-drawer";
import { usePlaygroundStore } from "../../store/playgroundStore";
import { executePipelineStream } from "../../features/playground/api";
import type { NodeType, BaseNodeData } from "../../types/pipeline";
import { useProjectState } from "../../hooks/queries/useProjectState";
import { useSaveProject } from "../../hooks/mutations/useSaveProject";
import { validatePipeline } from "../../utils/validation";
import { getFriendlyMessage, getErrorSuggestion } from "../../utils/errorFormatter";
import { MentorAssistant } from "../../features/mentor";
import { useAuthStore } from "../../store/authStore";
import { useMentorContext } from "../../features/mentor";
import { useAwardXP } from "../../features/gamification/hooks/useGamification";
import BadgeUnlockedToast from "../../features/gamification/components/BadgeUnlockedToast";
import LevelUpAnimation from "../../features/gamification/components/LevelUpAnimation";
import XPGainToast from "../../features/gamification/components/XPGainToast";
import { DatasetStoryPanel } from "../../components/playground/DatasetStoryPanel";
import { getStory } from "../../config/datasetStories";
import type { DatasetStory } from "../../config/datasetStories";
import {
  useMentorStore,
  type MentorAction,
} from "../../features/mentor/store/mentorStore";
import {
  Settings,
  ArrowLeft,
  BrainCircuit,
  GitBranch,
  BarChart3,
  Database,
  Layers,
} from "lucide-react";

// ─── Skeleton Loading Component ─────────────────────────────────

function PlaygroundSkeleton({ projectName }: { projectName?: string }) {
  return (
    <div className="h-screen flex flex-col bg-slate-50 overflow-hidden">
      {/* Toolbar skeleton */}
      <div className="h-16 bg-white/90 backdrop-blur-xl border-b border-slate-200/60 flex items-center justify-between px-6 shadow-lg shadow-slate-900/5">
        <div className="flex items-center gap-3">
          <div className="p-2 text-slate-300">
            <ArrowLeft className="w-5 h-5" />
          </div>
          <div className="flex items-center gap-3 mr-6">
            <div className="w-10 h-10 rounded-xl bg-slate-900 flex items-center justify-center shadow-lg shadow-slate-900/25">
              <Settings className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-slate-800">
                {projectName || "Loading project…"}
              </h1>
              <p className="text-xs text-slate-500">Visual Pipeline Builder</p>
            </div>
          </div>
          <div className="h-10 w-px bg-slate-200/60" />
          <div className="ml-3 w-36 h-10 bg-slate-200 rounded-lg animate-pulse" />
        </div>
        <div className="flex items-center gap-2">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="w-20 h-9 bg-slate-100 rounded-lg animate-pulse" />
          ))}
        </div>
      </div>

      {/* Main content skeleton */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar skeleton */}
        <div className="w-64 bg-white border-r border-slate-200/60 p-4 flex flex-col gap-4">
          <div className="h-10 bg-slate-100 rounded-lg animate-pulse" />
          <div className="space-y-2">
            <div className="h-3 w-20 bg-slate-200/60 rounded animate-pulse" />
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="h-10 bg-slate-100 rounded-lg animate-pulse"
                style={{ animationDelay: `${i * 150}ms` }}
              />
            ))}
          </div>
          <div className="space-y-2 mt-2">
            <div className="h-3 w-24 bg-slate-200/60 rounded animate-pulse" />
            {[1, 2, 3, 4].map((i) => (
              <div
                key={i}
                className="h-10 bg-slate-100 rounded-lg animate-pulse"
                style={{ animationDelay: `${(i + 3) * 150}ms` }}
              />
            ))}
          </div>
        </div>

        {/* Canvas skeleton — animated pipeline illustration */}
        <div className="flex-1 bg-slate-50 relative flex items-center justify-center">
          {/* Grid dots background */}
          <div
            className="absolute inset-0 opacity-30"
            style={{
              backgroundImage: "radial-gradient(circle, #94a3b8 1px, transparent 1px)",
              backgroundSize: "24px 24px",
            }}
          />

          {/* Animated loading content */}
          <div className="relative z-10 flex flex-col items-center">
            {/* Pipeline illustration */}
            <div className="flex items-center gap-3 mb-8">
              {[
                { icon: Database, color: "#3b82f6", label: "Data" },
                { icon: Layers, color: "#f59e0b", label: "Process" },
                { icon: GitBranch, color: "#06b6d4", label: "Split" },
                { icon: BrainCircuit, color: "#8b5cf6", label: "Model" },
                { icon: BarChart3, color: "#10b981", label: "Evaluate" },
              ].map((step, idx) => {
                const Icon = step.icon;
                return (
                  <div key={step.label} className="flex items-center gap-3">
                    <div className="flex flex-col items-center">
                      <div
                        className="w-14 h-14 rounded-xl flex items-center justify-center shadow-lg animate-pulse"
                        style={{
                          backgroundColor: `${step.color}15`,
                          border: `2px solid ${step.color}30`,
                          animationDelay: `${idx * 200}ms`,
                        }}
                      >
                        <Icon className="w-6 h-6" style={{ color: step.color }} />
                      </div>
                      <span className="text-[10px] text-slate-500 mt-1.5 font-medium">{step.label}</span>
                    </div>
                    {idx < 4 && (
                      <div className="flex gap-1 -mt-4">
                        {[0, 1, 2].map((dot) => (
                          <div
                            key={dot}
                            className="w-1.5 h-1.5 rounded-full bg-slate-300 animate-pulse"
                            style={{ animationDelay: `${idx * 200 + dot * 100}ms` }}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Loading text */}
            <div className="flex flex-col items-center gap-3">
              <div className="flex items-center gap-2">
                <div className="w-5 h-5 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                <p className="text-sm font-medium text-slate-600">Loading your pipeline…</p>
              </div>
              <p className="text-xs text-slate-500">Setting up workspace and restoring your progress</p>
            </div>

            {/* Progress shimmer bar */}
            <div className="mt-6 w-64 h-1.5 bg-slate-200 rounded-full overflow-hidden">
              <div className="h-full w-2/5 bg-gradient-to-r from-indigo-500 via-purple-400 to-indigo-500 rounded-full skeleton-shimmer" />
            </div>
          </div>

          <style>{`
            @keyframes skeletonShimmer {
              0% { transform: translateX(-100%); }
              50% { transform: translateX(200%); }
              100% { transform: translateX(-100%); }
            }
            .skeleton-shimmer { animation: skeletonShimmer 1.8s ease-in-out infinite; }
          `}</style>
        </div>
      </div>
    </div>
  );
}

// ─── Main Component ─────────────────────────────────────────────

export default function PlayGround() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [viewNodeId, setViewNodeId] = useState<string | null>(null);
  const [chatbotNodeId, setChatbotNodeId] = useState<string | null>(null);
  const [activityNodeId, setActivityNodeId] = useState<string | null>(null);
  const [resultsOpen, setResultsOpen] = useState(false);
  const [shareModalOpen, setShareModalOpen] = useState(false);
  const [exportModalOpen, setExportModalOpen] = useState(false);
  const [activeStory, setActiveStory] = useState<DatasetStory | null>(null);
  const streamCleanupRef = useRef<(() => void) | null>(null);
  const awardXP = useAwardXP();

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
    setNodeExecutionStatus,
    setAllNodesPending,
    clearExecutionStatus,
    animateEdgesForNode,
    addNodeResult,
    addExecutionLog,
    clearExecutionLogs,
  } = usePlaygroundStore();

  const { user } = useAuthStore();
  const { preferences, updatePreferences } = useMentorStore();

  const handleToggleMentor = () => {
    updatePreferences({ enabled: !preferences.enabled });
  };

  // Enable AI mentor context awareness
  useMentorContext({
    enabled: preferences.enabled,
    debounceMs: 1000,
  });

  // Handle node click - show view data modal for view nodes, config for others
  const handleNodeClick = (nodeId: string) => {
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return;

    // Activity nodes open the activity modal directly
    if (node.data.type.startsWith("activity_")) {
      setActivityNodeId(nodeId);
      return;
    }

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
  const { data: projectStateData, isLoading: isStateLoading } = useProjectState(projectId);
  const { data: projectData, isLoading: isProjectLoading } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => getProjectById(projectId!),
    enabled: !!projectId,
  });
  const saveProject = useSaveProject();

  const isPageLoading = isStateLoading || isProjectLoading;

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

  // Listen for custom event from Open Activity button on activity nodes
  useEffect(() => {
    const handleOpenActivityModal = (event: CustomEvent) => {
      const { nodeId } = event.detail;
      setActivityNodeId(nodeId);
    };

    window.addEventListener(
      "openActivityModal",
      handleOpenActivityModal as EventListener,
    );

    return () => {
      window.removeEventListener(
        "openActivityModal",
        handleOpenActivityModal as EventListener,
      );
    };
  }, []);

  // Listen for dataset story open event
  useEffect(() => {
    const handleOpenStory = (event: CustomEvent) => {
      const { datasetId } = event.detail;
      const story = getStory(datasetId);
      if (story) setActiveStory(story);
    };

    window.addEventListener("openDatasetStory", handleOpenStory as EventListener);
    return () => {
      window.removeEventListener("openDatasetStory", handleOpenStory as EventListener);
    };
  }, []);

  // Cleanup SSE stream on unmount only
  useEffect(() => {
    return () => {
      streamCleanupRef.current?.();
    };
  }, []);

  // Clear execution status when nodes or edges change (pipeline modified)
  useEffect(() => {
    clearExecutionStatus();
  }, [nodes.length, edges.length, clearExecutionStatus]);

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

    toast.success(`${template.label} template loaded!`);
  };

  const handleExecute = async () => {
    try {
      setIsExecuting(true);
      setResultsOpen(true);
      clearExecutionStatus();
      clearExecutionLogs();

      // 🛡️ VALIDATION: Check pipeline before execution
      const validationResult = validatePipeline(nodes, edges);

      if (!validationResult.isValid) {
        // Show validation errors as toast notifications
        validationResult.errors.forEach((error) => {
          const message = error.suggestion
            ? `${error.message}\n💡 ${error.suggestion}`
            : error.message;

          toast.error(message, { duration: 4000 });
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
            duration: 4000,
            icon: "⚠️",
            style: {
              border: "1px solid rgba(251, 191, 36, 0.3)",
            },
          });
        });
        setIsExecuting(false);
        return;
      }

      // Topological sort to determine execution order
      const sortedNodes = topologicalSort(nodes, edges);

      // Initialize all nodes as pending
      setAllNodesPending(sortedNodes.map((n) => n.id));

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
        label: node.data.label,
      }));

      console.log("🚀 Executing pipeline with streaming...", pipelineConfig);

      // Clear previous results and initialize empty result for streaming
      setExecutionResult({
        success: true,
        nodeResults: {},
        timestamp: new Date().toISOString(),
      });

      // Execute pipeline with streaming
      const cleanup = await executePipelineStream(
        {
          pipeline: pipelineConfig,
          edges: edges.map((edge) => ({
            source: edge.source,
            target: edge.target,
            sourceHandle: edge.sourceHandle || undefined,
            targetHandle: edge.targetHandle || undefined,
          })),
          pipeline_name: "Playground Pipeline",
        },
        {
          onNodeStarted: (event) => {
            console.log("🟡 Node started:", event);
            setNodeExecutionStatus(event.node_id, "running");
            addExecutionLog({
              timestamp: new Date().toISOString(),
              event: "node_started",
              nodeId: event.node_id,
              nodeLabel: event.label,
              message: `Started: ${event.label}`,
            });
          },
          onNodeCompleted: (event) => {
            console.log("🟢 Node completed:", event);
            setNodeExecutionStatus(event.node_id, "completed");
            animateEdgesForNode(event.node_id);

            // Incrementally add this node's result to the results panel
            addNodeResult(event.node_id, {
              success: true,
              output: event.result,
            });

            // Also store on the node data for view modals
            updateNode(event.node_id, { result: event.result });

            addExecutionLog({
              timestamp: new Date().toISOString(),
              event: "node_completed",
              nodeId: event.node_id,
              nodeLabel: event.label,
              message: `Completed: ${event.label}`,
            });
          },
          onNodeFailed: (event) => {
            console.log("🔴 Node failed:", event);
            setNodeExecutionStatus(event.node_id, "failed");

            // Incrementally add this node's error to the results panel
            addNodeResult(event.node_id, {
              success: false,
              error: event.error,
            });

            addExecutionLog({
              timestamp: new Date().toISOString(),
              event: "node_failed",
              nodeId: event.node_id,
              nodeLabel: event.label,
              message: `Failed: ${event.label} — ${getFriendlyMessage(event.error)}`,
            });
          },
          onPipelineCompleted: (event) => {
            console.log("✅ Pipeline completed:", event);

            // Finalize the execution result — keep the incrementally-built nodeResults
            const currentResult = usePlaygroundStore.getState().executionResult;
            setExecutionResult({
              ...currentResult,
              success: true,
              timestamp: new Date().toISOString(),
            });

            addExecutionLog({
              timestamp: new Date().toISOString(),
              event: "pipeline_completed",
              message: `Pipeline completed successfully (${event.nodes_executed} nodes)`,
            });

            setIsExecuting(false);
            streamCleanupRef.current = null;

            // Award XP for successful pipeline execution
            awardXP.mutate({ action: "pipeline_execution", context: { nodes_executed: event.nodes_executed } });
          },
          onPipelineFailed: (event) => {
            console.log("❌ Pipeline failed:", event);

            // Use structured error from backend (or fallback to string)
            const friendlyMsg = getFriendlyMessage(event.error) || "Pipeline execution failed";
            const suggestion = getErrorSuggestion(event.error) || undefined;

            // Keep incrementally-built nodeResults, add error info
            const currentResult = usePlaygroundStore.getState().executionResult;
            setExecutionResult({
              ...currentResult,
              success: false,
              error: friendlyMsg,
              errorSuggestion: suggestion,
              timestamp: new Date().toISOString(),
            });

            addExecutionLog({
              timestamp: new Date().toISOString(),
              event: "pipeline_failed",
              message: `Pipeline failed: ${friendlyMsg}`,
            });

            setIsExecuting(false);
            streamCleanupRef.current = null;
          },
          onError: (error) => {
            console.error("SSE Error:", error);
            toast.error("Connection error: " + error.message);
            setIsExecuting(false);
            streamCleanupRef.current = null;
          },
        },
      );

      // Store cleanup function for abort functionality
      streamCleanupRef.current = cleanup;
    } catch (error) {
      console.error("Pipeline execution failed:", error);
      setExecutionProgress(null);

      // Format user-friendly error message
      let userFriendlyError = "Pipeline execution failed";
      let errorDetails: Record<string, unknown> | undefined = undefined;
      let errorSuggestion: string | undefined = undefined;

      if (error && typeof error === "object" && "response" in error) {
        const responseError = (error as any).response?.data;
        const errorMsg =
          responseError?.detail ||
          responseError?.message ||
          responseError?.error;

        if (errorMsg) {
          userFriendlyError = errorMsg;
          if (responseError.details) {
            errorDetails = responseError.details;
          }
          if (responseError.suggestion) {
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

      setIsExecuting(false);
      streamCleanupRef.current = null;
    }
  };

  const handleAbort = () => {
    if (streamCleanupRef.current) {
      streamCleanupRef.current();
      streamCleanupRef.current = null;
    }
    setIsExecuting(false);
    clearExecutionStatus();
    toast.error("Pipeline execution aborted by user");
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
      toast.success(`Added ${nodeDef.label} to canvas`);
    } catch (error) {
      console.error("Error adding node from mentor:", error);
      toast.error("Failed to add node");
    }
  };

  // Handle execute pipeline action from mentor
  const handleExecutePipeline = () => {
    handleExecute();
  };

  // Show skeleton while initial data is loading
  if (isPageLoading) {
    return <PlaygroundSkeleton projectName={projectData?.name} />;
  }

  return (
    <div className="h-screen flex flex-col bg-slate-50">
      <Toaster
        position="top-right"
        gutter={10}
        containerStyle={{ top: 16, right: 16 }}
        toastOptions={{
          duration: 3000,
          style: {
            background: "rgba(15, 23, 42, 0.92)",
            backdropFilter: "blur(12px)",
            color: "#f1f5f9",
            borderRadius: "14px",
            padding: "14px 18px",
            fontSize: "13.5px",
            fontWeight: 500,
            lineHeight: "1.5",
            maxWidth: "420px",
            boxShadow:
              "0 20px 40px -12px rgba(0, 0, 0, 0.35), 0 0 0 1px rgba(148, 163, 184, 0.1)",
            border: "1px solid rgba(148, 163, 184, 0.15)",
          },
          success: {
            duration: 2500,
            iconTheme: { primary: "#34d399", secondary: "#0f172a" },
            style: {
              border: "1px solid rgba(52, 211, 153, 0.3)",
            },
          },
          error: {
            duration: 5000,
            iconTheme: { primary: "#f87171", secondary: "#0f172a" },
            style: {
              border: "1px solid rgba(248, 113, 113, 0.3)",
            },
          },
          loading: {
            iconTheme: { primary: "#94a3b8", secondary: "#0f172a" },
            style: {
              border: "1px solid rgba(148, 163, 184, 0.2)",
            },
          },
        }}
      />
      <Toolbar
        onExecute={handleExecute}
        onAbort={handleAbort}
        onClear={handleClear}
        onSave={handleSave}
        onExport={() => setExportModalOpen(true)}
        onShare={projectId ? handleShare : undefined}
        isExecuting={usePlaygroundStore.getState().isExecuting}
        isShareLoading={saveProject.isPending}
        executionProgress={executionProgress}
        projectName={projectData?.name}
        onBack={() => navigate("/dashboard")}
        isMentorEnabled={preferences.enabled}
        onToggleMentor={handleToggleMentor}
      />

      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 flex overflow-hidden">
          <ReactFlowProvider>
            <Sidebar
              onNodeDragStart={onNodeDragStart}
              onTemplateClick={handleTemplateClick}
            />
            <Canvas onNodeClick={handleNodeClick} />
          </ReactFlowProvider>
        </div>

        <ResultsDrawer
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

      <ActivityModal
        nodeId={activityNodeId}
        onClose={() => setActivityNodeId(null)}
      />

      <ShareModal
        isOpen={shareModalOpen}
        onClose={() => setShareModalOpen(false)}
        projectId={projectId ? parseInt(projectId) : 0}
        projectName={projectData?.name || "Untitled Project"}
      />

      <ExportModal
        isOpen={exportModalOpen}
        onClose={() => setExportModalOpen(false)}
      />

      <DatasetStoryPanel
        story={activeStory}
        onClose={() => setActiveStory(null)}
      />

      <BadgeUnlockedToast />
      <LevelUpAnimation />
      <XPGainToast />

      <MentorAssistant
        userName={user?.fullName || user?.emailId || "there"}
        onAction={(action: MentorAction) => {
          console.log("[Mentor] Action clicked:", action);

          if (action.type === "select_algorithm") {
            const algorithm = action.payload?.algorithm as string;
            if (algorithm) {
              const store = useMentorStore.getState();
              store.setSelectedAlgorithm(algorithm);
              store.setStage("algorithm_selected");
            }
          } else if (action.type === "add_node") {
            const nodeType = (action.payload?.node_type ||
              action.payload?.model_type) as string;
            if (nodeType) {
              handleAddNodeFromMentor(nodeType);
            }
          } else if (action.type === "execute") {
            handleExecutePipeline();
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
