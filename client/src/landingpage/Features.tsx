import React from "react";
import { motion } from "framer-motion";
import {
  Workflow,
  Sparkles,
  Palette,
  Database,
  BarChart3,
  Share2,
} from "lucide-react";

const Features: React.FC = () => {
  const features = [
    {
      icon: Workflow,
      title: "Visual ML Pipelines",
      description:
        "Build complex machine learning workflows with an intuitive drag-and-drop interface.",
    },
    {
      icon: Sparkles,
      title: "GenAI Workflows",
      description:
        "Leverage AI assistance for node suggestions, validation, and optimization.",
    },
    {
      icon: Palette,
      title: "Custom UI Builder",
      description:
        "Export production-ready user interfaces for your ML models automatically.",
    },
    {
      icon: Database,
      title: "Dataset & Preprocessing",
      description:
        "Import, clean, and transform your data with powerful preprocessing tools.",
    },
    {
      icon: BarChart3,
      title: "Training & Metrics",
      description:
        "Train models and visualize performance with comprehensive metrics and charts.",
    },
    {
      icon: Share2,
      title: "Shareable Pipelines",
      description:
        "Collaborate seamlessly by sharing pipelines with your team or community.",
    },
  ];

  return (
    <section id="features" className="py-24 px-6 lg:px-8 bg-white">
      <div className="max-w-7xl mx-auto">
        <motion.div
          className="text-center space-y-4 mb-16"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-4xl lg:text-5xl font-bold text-gray-900">
            Everything you need to ship ML products
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            From data ingestion to deployment, Visual ML provides a complete
            toolkit for building production-ready machine learning solutions.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <motion.div
                key={index}
                className="space-y-4"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                whileHover={{ scale: 1.02, transition: { duration: 0.2 } }}
              >
                <div className="w-12 h-12 bg-gray-100 rounded-lg flex items-center justify-center">
                  <Icon className="w-6 h-6 text-gray-900" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900">
                  {feature.title}
                </h3>
                <p className="text-gray-600 leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default Features;
