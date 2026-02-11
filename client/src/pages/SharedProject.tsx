import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type Node,
  type Edge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Copy, ArrowLeft, Eye, Users, Lock } from "lucide-react";
import MLNode from "../components/playground/MLNode";

// Use the same node type mapping as Canvas
const nodeTypes: Record<string, React.ComponentType<any>> = {
  upload_file: MLNode,
  select_dataset: MLNode,
  sample_dataset: MLNode,
  table_view: MLNode,
  data_preview: MLNode,
  statistics_view: MLNode,
  column_info: MLNode,
  chart_view: MLNode,
  missing_value_handler: MLNode,
  encoding: MLNode,
  transformation: MLNode,
  split: MLNode,
  preprocess: MLNode,
  feature_selection: MLNode,
  scaling: MLNode,
  train: MLNode,
  evaluate: MLNode,
  linear_regression: MLNode,
  logistic_regression: MLNode,
  decision_tree: MLNode,
  random_forest: MLNode,
  r2_score: MLNode,
  mse_score: MLNode,
  rmse_score: MLNode,
  mae_score: MLNode,
  confusion_matrix: MLNode,
  classification_report: MLNode,
  accuracy_score: MLNode,
  precision_score: MLNode,
  recall_score: MLNode,
  f1_score: MLNode,
  roc_curve: MLNode,
  precision_recall_curve: MLNode,
  "genai/llm": MLNode,
  "genai/embeddings": MLNode,
  "genai/vector_store": MLNode,
  "genai/retriever": MLNode,
  "genai/prompt": MLNode,
  "genai/chain": MLNode,
  chatbot_node: MLNode,
  "genai/conditional": MLNode,
  "genai/loop": MLNode,
  "genai/function": MLNode,
  "genai/api": MLNode,
};

interface ProjectData {
  id: number;
  name: string;
  description?: string;
  owner: {
    id: number;
    fullName: string;
    emailId: string;
  };
  nodes: Node[];
  edges: Edge[];
  allow_cloning: boolean;
  view_count: number;
  clone_count: number;
  created_at: string;
  updated_at: string;
}

export const SharedProject = () => {
  const { shareToken } = useParams<{ shareToken: string }>();
  const navigate = useNavigate();
  const user = localStorage.getItem("user")
    ? JSON.parse(localStorage.getItem("user")!)
    : null;
  const [project, setProject] = useState<ProjectData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isCloning, setIsCloning] = useState(false);
  const [cloneSuccess, setCloneSuccess] = useState(false);

  useEffect(() => {
    if (shareToken) {
      fetchSharedProject();
    }
  }, [shareToken]);

  const fetchSharedProject = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(
        `${import.meta.env.VITE_API_URL}/api/v1/projects/shared/${shareToken}`,
        {
          credentials: "include",
        },
      );

      if (!response.ok) {
        if (response.status === 404) {
          throw new Error("Project not found or no longer shared");
        }
        throw new Error("Failed to load shared project");
      }

      const data = await response.json();
      setProject(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  const handleClone = async () => {
    if (!user) {
      // Redirect to login
      navigate("/login", { state: { from: `/shared/${shareToken}` } });
      return;
    }

    try {
      setIsCloning(true);
      const response = await fetch(
        `${import.meta.env.VITE_API_URL}/api/v1/projects/shared/${shareToken}/clone`,
        {
          method: "POST",
          credentials: "include",
        },
      );

      if (!response.ok) {
        throw new Error("Failed to clone project");
      }

      const data = await response.json();
      setCloneSuccess(true);

      // Redirect to the cloned project after a short delay
      setTimeout(() => {
        navigate(`/playground/${data.project_id}`);
      }, 1500);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to clone project");
    } finally {
      setIsCloning(false);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-slate-600 font-medium">
            Loading shared project...
          </p>
        </div>
      </div>
    );
  }

  if (error || !project) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
        <div className="bg-white rounded-2xl shadow-xl p-8 max-w-md text-center">
          <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <Lock className="w-8 h-8 text-red-600" />
          </div>
          <h2 className="text-2xl font-bold text-slate-800 mb-2">
            Unable to Load Project
          </h2>
          <p className="text-slate-600 mb-6">
            {error || "This project may have been unshared or deleted."}
          </p>
          <button
            onClick={() => navigate("/")}
            className="px-6 py-2 bg-slate-900 text-white rounded-lg hover:bg-slate-800 transition-colors"
          >
            Go to Home
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header Banner */}
      <div className="bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => navigate("/")}
                className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <ArrowLeft className="w-5 h-5 text-slate-600" />
              </button>
              <div>
                <h1 className="text-xl font-bold text-slate-800">
                  {project.name}
                </h1>
                <p className="text-sm text-slate-500">
                  Shared by {project.owner.fullName}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              {/* Stats */}
              <div className="flex items-center gap-4 px-4 py-2 bg-slate-50 rounded-lg border border-slate-200">
                <div className="flex items-center gap-2">
                  <Eye className="w-4 h-4 text-slate-500" />
                  <span className="text-sm font-medium text-slate-700">
                    {project.view_count}
                  </span>
                </div>
                <div className="w-px h-4 bg-slate-300"></div>
                <div className="flex items-center gap-2">
                  <Users className="w-4 h-4 text-slate-500" />
                  <span className="text-sm font-medium text-slate-700">
                    {project.clone_count} clones
                  </span>
                </div>
              </div>

              {/* Clone Button */}
              {project.allow_cloning && (
                <button
                  onClick={handleClone}
                  disabled={isCloning || cloneSuccess}
                  className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-green-600 text-white rounded-lg flex items-center gap-2 transition-all shadow-lg hover:shadow-xl font-semibold disabled:cursor-default"
                >
                  {cloneSuccess ? (
                    <>✓ Cloned! Redirecting...</>
                  ) : isCloning ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                      Cloning...
                    </>
                  ) : (
                    <>
                      <Copy className="w-4 h-4" />
                      Clone to My Projects
                    </>
                  )}
                </button>
              )}
            </div>
          </div>

          {/* Description */}
          {project.description && (
            <p className="mt-3 text-sm text-slate-600 max-w-3xl">
              {project.description}
            </p>
          )}
        </div>
      </div>

      {/* Read-only Pipeline Canvas */}
      <div className="h-[calc(100vh-140px)]">
        <ReactFlow
          nodes={project.nodes}
          edges={project.edges}
          nodeTypes={nodeTypes}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          zoomOnScroll={true}
          panOnScroll={true}
          fitView
        >
          <Background />
          <Controls />
          <MiniMap />
        </ReactFlow>
      </div>

      {/* Read-only Notice */}
      <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 bg-amber-50 border border-amber-200 rounded-lg px-6 py-3 shadow-lg">
        <div className="flex items-center gap-2">
          <Eye className="w-4 h-4 text-amber-700" />
          <p className="text-sm font-medium text-amber-800">
            Viewing in read-only mode
            {project.allow_cloning && (
              <span className="ml-1">
                — Click "Clone" to edit your own copy
              </span>
            )}
          </p>
        </div>
      </div>
    </div>
  );
};
