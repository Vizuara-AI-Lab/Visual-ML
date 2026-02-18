import React from "react";
import { ArrowRight, Code2, Sparkles } from "lucide-react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router";
import WorkflowCanvas from "./WorkflowCanvas";

const Hero: React.FC = () => {
  const navigate = useNavigate();

  return (
    <section className="relative min-h-screen bg-white overflow-hidden">
      {/* Grid pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#f0f0f0_1px,transparent_1px),linear-gradient(to_bottom,#f0f0f0_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_110%)]" />

      {/* Gradient orbs */}
      <div className="absolute top-20 left-1/4 w-96 h-96 bg-linear-to-br from-blue-400/10 to-violet-400/10 rounded-full blur-3xl animate-pulse" />
      <div
        className="absolute top-40 right-1/4 w-80 h-80 bg-linear-to-br from-violet-400/10 to-pink-400/10 rounded-full blur-3xl animate-pulse"
        style={{ animationDelay: "1.5s" }}
      />

      <div className="relative max-w-7xl mx-auto px-6 lg:px-8">
        <div className="pt-32 pb-20 lg:pt-40 lg:pb-28">
          {/* Main Content */}
          <div className="max-w-4xl mx-auto text-center space-y-8">
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.05 }}
              className="flex justify-center"
            >
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-slate-100/80 backdrop-blur-sm rounded-full border border-slate-200/60">
                <Sparkles className="w-4 h-4 text-slate-600" />
                <span className="text-sm font-medium text-slate-600">
                  Now with GenAI Integration
                </span>
                <span className="px-2 py-0.5 text-[10px] font-semibold text-emerald-700 bg-emerald-100 rounded-full border border-emerald-200">
                  New
                </span>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="space-y-6"
            >
              <h1 className="text-6xl lg:text-7xl xl:text-8xl font-bold tracking-tight">
                <span className="block bg-linear-to-br from-slate-900 via-slate-800 to-slate-900 bg-clip-text text-transparent">
                  Machine Learning
                </span>
                <span className="block mt-2 bg-linear-to-r from-slate-400 via-slate-300 to-slate-400 bg-clip-text text-transparent">
                  Made Visual
                </span>
              </h1>
              <p className="text-xl lg:text-2xl text-slate-600 max-w-3xl mx-auto leading-relaxed font-light">
                Build production-ready ML pipelines with an intuitive
                drag-and-drop interface. Enterprise-grade performance, zero
                code.
              </p>
            </motion.div>

            {/* CTA Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="flex flex-col sm:flex-row gap-4 justify-center items-center pt-4"
            >
              <button
                onClick={() => navigate("/signup")}
                className="group relative px-8 py-4 bg-slate-900 text-white rounded-xl font-medium hover:bg-slate-800 transition-all duration-300 flex items-center gap-2 shadow-lg shadow-slate-900/25 hover:shadow-xl hover:shadow-slate-900/40 hover:-translate-y-0.5"
              >
                <span>Get Started Free</span>
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </button>
              <button className="px-8 py-4 text-slate-700 font-medium hover:text-slate-900 transition-colors flex items-center gap-2 bg-white/50 backdrop-blur-sm rounded-xl border border-slate-200/60 hover:bg-white/80 hover:border-slate-300/60">
                <Code2 className="w-4 h-4" />
                <span>Documentation</span>
              </button>
            </motion.div>
          </div>

          {/* Visual Workflow Preview */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="mt-24 max-w-6xl mx-auto relative"
          >
            <WorkflowCanvas />
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
