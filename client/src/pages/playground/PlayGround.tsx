import { useState, useEffect } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { useParams } from "react-router";
import "@xyflow/react/dist/style.css";
import { Sidebar } from "../../components/playground/Sidebar";
import { Canvas } from "../../components/playground/Canvas";
import { ConfigModal } from "../../components/playground/ConfigModal";
import { Toolbar } from "../../components/playground/Toolbar";
import { ResultsPanel } from "../../components/playground/ResultsPanel";
import { usePlaygroundStore } from "../../store/playgroundStore";
import { executePipeline } from "../../features/playground/api";
import type { NodeType, BaseNodeData } from "../../types/pipeline";
import { useProjectState } from "../../hooks/queries/useProjectState";
import { useSaveProject } from "../../hooks/mutations/useSaveProject";

export default function PlayGround() {
  const { projectId } = useParams<{ projectId: string }>();
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [resultsOpen, setResultsOpen] = useState(false);
  
  const {
    nodes,
    edges,
    setExecutionResult,
    setIsExecuting,
    clearAll,
    setCurrentProjectId,
    loadProjectState,
    getProjectState,
  } = usePlaygroundStore();

  // Load project state if projectId exists
  const { data: projectStateData } = useProjectState(projectId);
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

  // Auto-save every 5 seconds when there are changes
  useEffect(() => {
    if (!projectId || nodes.length === 0) return;

    const saveTimer = setTimeout(() => {
      const state = getProjectState();
      saveProject.mutate({ id: projectId, state });
    }, 5000); // 5 seconds debounce

    return () => clearTimeout(saveTimer);
  }, [nodes, edges, projectId, saveProject, getProjectState]);

  const onNodeDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData("application/reactflow", nodeType);
    event.dataTransfer.effectAllowed = "move";
  };

  const handleExecute = async () => {
    try {
      setIsExecuting(true);
      setResultsOpen(true);

      // Topological sort to determine execution order
      const sortedNodes = topologicalSort(nodes, edges);

      const pipelineConfig = sortedNodes.map((node) => ({
        node_id: node.id,
        node_type: node.type as NodeType,
        input: node.data.config || {},
      }));

      const result = await executePipeline({
        pipeline: pipelineConfig,
        pipeline_name: "Playground Pipeline",
      });

      setExecutionResult({
        success: result.success,
        nodeResults: {},
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
    saveProject.mutate({ id: projectId, state });
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
      />

      <div className="flex-1 flex overflow-hidden">
        <ReactFlowProvider>
          <Sidebar onNodeDragStart={onNodeDragStart} />
          <Canvas onNodeClick={setSelectedNodeId} />
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
