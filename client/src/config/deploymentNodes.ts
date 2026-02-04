/**
 * Deployment Node Definitions
 */

import type { NodeCategory } from "./nodeDefinitions";
import { Cloud, Zap } from "lucide-react";

export const deploymentCategory: NodeCategory = {
  id: "deployment",
  label: "Deployment",
  icon: Cloud,
  nodes: [
    {
      type: "model_export",
      label: "Export Model",
      description: "Export trained model for deployment",
      category: "deployment",
      icon: Zap,
      color: "#10B981",
      defaultConfig: {
        format: "pickle",
        model_path: "",
      },
    },
    {
      type: "api_endpoint",
      label: "Create API",
      description: "Deploy model as REST API",
      category: "deployment",
      icon: Cloud,
      color: "#06B6D4",
      defaultConfig: {
        endpoint_name: "",
        model_path: "",
      },
    },
  ],
};
