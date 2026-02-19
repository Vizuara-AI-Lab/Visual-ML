/**
 * Activity Modal — fullscreen interactive learning activity viewer
 */

import { lazy, Suspense, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Loader2, Maximize2, Minimize2 } from "lucide-react";
import { usePlaygroundStore } from "../../../store/playgroundStore";
import { getNodeByType } from "../../../config/nodeDefinitions";

// Beginner: Foundations
const LossFunctionsActivity = lazy(() => import("./LossFunctionsActivity"));
const LinearRegressionActivity = lazy(() => import("./LinearRegressionActivity"));
const GradientDescentActivity = lazy(() => import("./GradientDescentActivity"));
const LogisticRegressionActivity = lazy(() => import("./LogisticRegressionActivity"));
const KNNPlaygroundActivity = lazy(() => import("./KNNPlaygroundActivity"));
const KMeansClusteringActivity = lazy(() => import("./KMeansClusteringActivity"));

// Intermediate: Model Understanding
const DecisionTreeActivity = lazy(() => import("./DecisionTreeActivity"));
const ConfusionMatrixActivity = lazy(() => import("./ConfusionMatrixActivity"));

// Advanced: Deep Learning
const ActivationFunctionsActivity = lazy(() => import("./ActivationFunctionsActivity"));
const NeuralNetworkActivity = lazy(() => import("./NeuralNetworkActivity"));
const BackpropagationActivity = lazy(() => import("./BackpropagationActivity"));
const CNNFiltersActivity = lazy(() => import("./CNNFiltersActivity"));
const OverfittingActivity = lazy(() => import("./OverfittingActivity"));

interface ActivityModalProps {
  nodeId: string | null;
  onClose: () => void;
}

function ActivityFallback() {
  return (
    <div className="flex-1 flex items-center justify-center">
      <div className="flex flex-col items-center gap-3">
        <Loader2 className="w-8 h-8 text-violet-500 animate-spin" />
        <p className="text-sm text-slate-500 font-medium">
          Loading activity...
        </p>
      </div>
    </div>
  );
}

const ACTIVITY_MAP: Record<string, React.LazyExoticComponent<() => JSX.Element>> = {
  activity_loss_functions: LossFunctionsActivity,
  activity_linear_regression: LinearRegressionActivity,
  activity_gradient_descent: GradientDescentActivity,
  activity_logistic_regression: LogisticRegressionActivity,
  activity_knn_playground: KNNPlaygroundActivity,
  activity_kmeans_clustering: KMeansClusteringActivity,
  activity_decision_tree: DecisionTreeActivity,
  activity_confusion_matrix: ConfusionMatrixActivity,
  activity_activation_functions: ActivationFunctionsActivity,
  activity_neural_network: NeuralNetworkActivity,
  activity_backpropagation: BackpropagationActivity,
  activity_cnn_filters: CNNFiltersActivity,
  activity_overfitting: OverfittingActivity,
};

export const ActivityModal = ({ nodeId, onClose }: ActivityModalProps) => {
  const { nodes } = usePlaygroundStore();
  const [isFullscreen, setIsFullscreen] = useState(false);

  const node = nodeId ? nodes.find((n) => n.id === nodeId) : null;
  const nodeDef = node ? getNodeByType(node.data.type) : null;

  if (!nodeId || !node || !nodeDef) return null;

  const nodeType = node.data.type;
  const accentColor = nodeDef.color;
  const Icon = nodeDef.icon;

  const ActivityComponent = ACTIVITY_MAP[nodeType];

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center">
        {/* Backdrop */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 bg-black/40 backdrop-blur-sm"
          onClick={onClose}
        />

        {/* Modal */}
        <motion.div
          initial={{ scale: 0.95, opacity: 0, y: 20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.95, opacity: 0, y: 20 }}
          transition={{ type: "spring", damping: 25, stiffness: 300 }}
          className={`relative bg-white border border-slate-200 shadow-2xl flex flex-col overflow-hidden z-10 transition-all duration-300 ${
            isFullscreen
              ? "w-full h-full rounded-none"
              : "w-11/12 h-5/6 rounded-xl"
          }`}
        >
          {/* Header */}
          <div className="px-6 py-3 border-b border-slate-200 flex items-center justify-between shrink-0">
            <div className="flex items-center gap-3">
              <div
                className="w-9 h-9 rounded-lg flex items-center justify-center"
                style={{ backgroundColor: `${accentColor}18` }}
              >
                <Icon className="w-5 h-5" style={{ color: accentColor }} />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-slate-800">
                  {nodeDef.label}
                </h2>
                <p className="text-xs text-slate-500">{nodeDef.description}</p>
              </div>
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={() => setIsFullscreen((f) => !f)}
                className="p-2 rounded-lg hover:bg-slate-100 transition-colors"
                title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
              >
                {isFullscreen ? (
                  <Minimize2 className="w-5 h-5 text-slate-500" />
                ) : (
                  <Maximize2 className="w-5 h-5 text-slate-500" />
                )}
              </button>
              <button
                onClick={onClose}
                className="p-2 rounded-lg hover:bg-slate-100 transition-colors"
              >
                <X className="w-5 h-5 text-slate-500" />
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-auto p-6">
            <Suspense fallback={<ActivityFallback />}>
              {ActivityComponent ? (
                <ActivityComponent />
              ) : (
                <div className="flex-1 flex items-center justify-center">
                  <p className="text-slate-500">Unknown activity type</p>
                </div>
              )}
            </Suspense>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};
