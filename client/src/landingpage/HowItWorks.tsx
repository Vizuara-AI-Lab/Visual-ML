import React from "react";
import { motion } from "framer-motion";
import { MousePointer2, Sparkles, Rocket } from "lucide-react";

const HowItWorks: React.FC = () => {
  const steps = [
    {
      icon: MousePointer2,
      title: "Drag and drop nodes",
      description:
        "Select from our library of ML nodes and arrange them visually to create your pipeline.",
    },
    {
      icon: Sparkles,
      title: "Let GenAI assist",
      description:
        "Get intelligent suggestions for optimal configurations, connections, and best practices.",
    },
    {
      icon: Rocket,
      title: "Deploy and share",
      description:
        "Export your custom UI, deploy your model, and share your pipeline with your team.",
    },
  ];

  return (
    <section id="how-it-works" className="py-24 px-6 lg:px-8 bg-gray-50">
      <div className="max-w-7xl mx-auto">
        <motion.div
          className="text-center space-y-4 mb-16"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-4xl lg:text-5xl font-bold text-gray-900">
            How it works
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Three simple steps to build and deploy production-ready ML
            workflows.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-12">
          {steps.map((step, index) => {
            const Icon = step.icon;
            return (
              <motion.div
                key={index}
                className="text-center space-y-4"
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
                whileHover={{ y: -8, transition: { duration: 0.3 } }}
              >
                <motion.div
                  className="inline-flex w-16 h-16 bg-gray-900 text-white rounded-xl items-center justify-center mb-2"
                  whileHover={{ scale: 1.1, rotate: 5 }}
                  transition={{ duration: 0.3 }}
                >
                  <Icon className="w-8 h-8" />
                </motion.div>
                <div className="space-y-2">
                  <div className="text-sm font-semibold text-gray-500">
                    Step {index + 1}
                  </div>
                  <h3 className="text-2xl font-semibold text-gray-900">
                    {step.title}
                  </h3>
                  <p className="text-gray-600 leading-relaxed">
                    {step.description}
                  </p>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
