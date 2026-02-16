import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Workflow, Sparkles, Zap } from "lucide-react";

const Features: React.FC = () => {
  const [activeSection, setActiveSection] = useState(0);
  const sectionRefs = [
    useRef<HTMLDivElement>(null),
    useRef<HTMLDivElement>(null),
    useRef<HTMLDivElement>(null),
  ];

  const features = [
    {
      id: 0,
      icon: Workflow,
      badge: "Visual Builder",
      title: "Design ML Workflows Visually",
      description:
        "Build complex machine learning pipelines with an intuitive drag-and-drop interface. No coding required. Connect nodes, configure parameters, and watch your models come to life in real-time.",
      highlights: [
        "Drag-and-drop interface",
        "Real-time validation",
        "Auto-save & version control",
        "Pre-built templates",
      ],
      gradient: "from-blue-500/20 to-violet-500/20",
    },
    {
      id: 1,
      icon: Sparkles,
      badge: "GenAI Powered",
      title: "AI Assistant That Understands ML",
      description:
        "Get intelligent suggestions for node configurations, optimal hyperparameters, and architecture recommendations. Our GenAI engine learns from your data and guides you to better models faster.",
      highlights: [
        "Smart node suggestions",
        "Hyperparameter optimization",
        "Architecture recommendations",
        "Automated code generation",
      ],
      gradient: "from-violet-500/20 to-purple-500/20",
    },
    {
      id: 2,
      icon: Zap,
      badge: "Production Ready",
      title: "Deploy With One Click",
      description:
        "Transform your ML pipelines into production-ready APIs instantly. Auto-generate documentation, monitoring dashboards, and scalable infrastructure. From prototype to production in minutes.",
      highlights: ["One-click deployment", "Built-in monitoring"],
      gradient: "from-purple-500/20 to-pink-500/20",
    },
  ];

  useEffect(() => {
    const observers = sectionRefs.map((ref, index) => {
      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              setActiveSection(index);
            }
          });
        },
        {
          threshold: 0.5,
          rootMargin: "-20% 0px -20% 0px",
        },
      );

      if (ref.current) {
        observer.observe(ref.current);
      }

      return observer;
    });

    return () => {
      observers.forEach((observer) => observer.disconnect());
    };
  }, []);

  const activeFeature = features[activeSection];

  return (
    <section id="features" className="relative bg-slate-50">
      {/* Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#e5e7eb_1px,transparent_1px),linear-gradient(to_bottom,#e5e7eb_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-40" />
      </div>

      <div className="relative max-w-7xl mx-auto px-6 lg:px-8 py-32">
        {/* Section Header */}
        <motion.div
          className="text-center space-y-4 mb-20"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-4xl lg:text-5xl font-bold bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 bg-clip-text text-transparent">
            Everything you need to ship ML products
          </h2>
          <p className="text-xl text-slate-600 max-w-2xl mx-auto font-light">
            From design to deployment, Visual ML provides the complete toolkit
          </p>
        </motion.div>

        {/* Sticky Scroll Section */}
        <div className="grid lg:grid-cols-2 gap-16">
          {/* Left Side - Sticky Content */}
          <div
            style={{ position: "sticky", top: "6rem", alignSelf: "flex-start" }}
            className="hidden lg:block space-y-8"
          >
            <AnimatePresence mode="wait">
              <motion.div
                key={activeSection}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.5 }}
                className="space-y-6"
              >
                {/* Badge */}
                <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/80 backdrop-blur-xl border border-slate-200/60 rounded-full">
                  <div
                    className={`w-2 h-2 rounded-full bg-linear-to-r ${activeFeature.gradient.replace("/20", "")}`}
                  />
                  <span className="text-sm font-medium text-slate-700">
                    {activeFeature.badge}
                  </span>
                </div>

                {/* Title */}
                <h3 className="text-4xl lg:text-5xl font-bold text-slate-900 leading-tight">
                  {activeFeature.title}
                </h3>

                {/* Description */}
                <p className="text-lg text-slate-600 leading-relaxed">
                  {activeFeature.description}
                </p>

                {/* Highlights */}
                <div className="space-y-3">
                  {activeFeature.highlights.map((highlight, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.3, delay: index * 0.1 }}
                      className="flex items-center gap-3"
                    >
                      <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-slate-900 to-slate-700 flex items-center justify-center flex-shrink-0">
                        <svg
                          className="w-4 h-4 text-white"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M5 13l4 4L19 7"
                          />
                        </svg>
                      </div>
                      <span className="text-slate-700 font-medium">
                        {highlight}
                      </span>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Right Side - Scrollable Sections */}
          <div className="space-y-16 lg:space-y-96 lg:col-start-2">
            {features.map((feature, index) => (
              <div
                key={feature.id}
                ref={sectionRefs[index]}
                className="min-h-[400px] flex items-center"
              >
                <motion.div
                  initial={{ opacity: 0, x: 50 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.6 }}
                  className={`p-8 rounded-3xl border-2 transition-all duration-500 ${
                    activeSection === index
                      ? "bg-white border-slate-900 shadow-2xl shadow-slate-900/10"
                      : "bg-white/40 border-slate-200/60 shadow-lg"
                  }`}
                >
                  <div className="flex items-start gap-4 mb-4">
                    <div
                      className={`w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-500 ${
                        activeSection === index
                          ? "bg-gradient-to-br from-slate-900 to-slate-700 shadow-lg shadow-slate-900/25"
                          : "bg-slate-100"
                      }`}
                    >
                      {React.createElement(feature.icon, {
                        className: `w-6 h-6 transition-colors duration-500 ${
                          activeSection === index
                            ? "text-white"
                            : "text-slate-400"
                        }`,
                      })}
                    </div>
                    <div className="flex-1">
                      <h4
                        className={`text-2xl font-bold mb-2 transition-colors duration-500 ${
                          activeSection === index
                            ? "text-slate-900"
                            : "text-slate-400"
                        }`}
                      >
                        {feature.title}
                      </h4>
                      <p
                        className={`text-base leading-relaxed transition-colors duration-500 ${
                          activeSection === index
                            ? "text-slate-600"
                            : "text-slate-400"
                        }`}
                      >
                        {feature.description}
                      </p>
                    </div>
                  </div>
                </motion.div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default Features;
