import React from "react";
import { motion } from "framer-motion";
import { Lightbulb, Shield, TrendingUp, Cpu } from "lucide-react";

const GenAISection: React.FC = () => {
  const features = [
    {
      icon: Lightbulb,
      title: "Smart Suggestions",
      description:
        "GenAI recommends optimal nodes and configurations as you build your pipeline.",
    },
    {
      icon: Shield,
      title: "Auto Validation",
      description:
        "Prevent errors before they happen with intelligent validation and best practice enforcement.",
    },
    {
      icon: TrendingUp,
      title: "Explainable Insights",
      description:
        "Understand model behavior with AI-generated explanations and performance insights.",
    },
    {
      icon: Cpu,
      title: "Hardware Integration",
      description:
        "Seamless hardware integration for faster experiments and real-time performance.",
    },
  ];

  return (
    <section id="genai" className="py-24 px-6 lg:px-8 bg-gray-50">
      <div className="max-w-7xl mx-auto">
        <motion.div
          className="text-center space-y-4 mb-16"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-4xl lg:text-5xl font-bold text-gray-900">
            GenAI that accelerates every workflow
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Build smarter, faster, and with confidence. Our GenAI engine assists
            you at every step, from node recommendations to deployment
            optimization.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <motion.div
                key={index}
                className="bg-white border border-gray-200 rounded-xl p-6 hover:shadow-lg transition-shadow"
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                whileHover={{
                  y: -8,
                  scale: 1.03,
                  transition: { duration: 0.3 },
                }}
              >
                <motion.div
                  className="w-12 h-12 bg-gray-100 rounded-lg flex items-center justify-center mb-4"
                  whileHover={{ rotate: 360, scale: 1.1 }}
                  transition={{ duration: 0.5 }}
                >
                  <Icon className="w-6 h-6 text-gray-900" />
                </motion.div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  {feature.title}
                </h3>
                <p className="text-sm text-gray-600 leading-relaxed">
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

export default GenAISection;
