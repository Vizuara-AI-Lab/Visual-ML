import React from "react";
import { Sparkles, Workflow, Share2 } from "lucide-react";
import NodeCanvas from "./NodeCanvas";

const Hero: React.FC = () => {
  const highlights = [
    { icon: Sparkles, text: "GenAI workflows" },
    { icon: Workflow, text: "Custom UI builder" },
    { icon: Share2, text: "Shareable pipelines" },
  ];

  return (
    <section className="relative pt-32 pb-20 px-6 lg:px-8 overflow-hidden">
      <NodeCanvas />
      <div className="max-w-7xl mx-auto relative z-10">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          <div className="space-y-8">
            <div className="space-y-6">
              <h1 className="text-5xl lg:text-6xl font-bold text-gray-900 leading-tight">
                Build ML workflows visually.
              </h1>
              <p className="text-xl text-gray-600 leading-relaxed">
                Create powerful machine learning pipelines without code.
                Leverage GenAI assistance to build, validate, and deploy
                workflows. Export custom UIs and share your work instantly.
              </p>
            </div>

            <div className="flex flex-wrap gap-4">
              <button className="px-6 py-3 bg-gray-900 text-white font-medium rounded-lg hover:bg-gray-800 transition-colors shadow-sm">
                Try the builder
              </button>
              <button className="px-6 py-3 bg-white text-gray-900 font-medium rounded-lg border border-gray-300 hover:border-gray-400 transition-colors">
                Explore templates
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4">
              {highlights.map((item, index) => (
                <div key={index} className="flex items-center space-x-3">
                  <div className="flex-shrink-0 w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center">
                    <item.icon className="w-5 h-5 text-gray-900" />
                  </div>
                  <span className="text-sm font-medium text-gray-900">
                    {item.text}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="lg:pl-8">{/* Canvas is now in background */}</div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
