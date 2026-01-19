import React from "react";
import { FileText, Image, TrendingUp, MessageSquare } from "lucide-react";

const Templates: React.FC = () => {
  const templates = [
    {
      icon: FileText,
      title: "Text Classification",
      description:
        "Build sentiment analysis and document categorization pipelines.",
      tags: ["NLP", "Classification"],
    },
    {
      icon: Image,
      title: "Image Recognition",
      description:
        "Create computer vision models for object detection and image classification.",
      tags: ["Vision", "CNN"],
    },
    {
      icon: TrendingUp,
      title: "Time Series Forecasting",
      description:
        "Predict future values from historical data with regression models.",
      tags: ["Forecasting", "Regression"],
    },
    {
      icon: MessageSquare,
      title: "GenAI Chatbot",
      description:
        "Deploy conversational AI with LLM integration and custom prompts.",
      tags: ["GenAI", "LLM"],
    },
  ];

  return (
    <section id="templates" className="py-24 px-6 lg:px-8 bg-white">
      <div className="max-w-7xl mx-auto">
        <div className="text-center space-y-4 mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-gray-900">
            Start with templates
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Get started quickly with pre-built templates for common ML use
            cases. Customize and extend them to fit your needs.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {templates.map((template, index) => {
            const Icon = template.icon;
            return (
              <div
                key={index}
                className="bg-white border border-gray-200 rounded-xl p-6 hover:shadow-lg hover:border-gray-300 transition-all cursor-pointer"
              >
                <div className="w-12 h-12 bg-gray-100 rounded-lg flex items-center justify-center mb-4">
                  <Icon className="w-6 h-6 text-gray-900" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  {template.title}
                </h3>
                <p className="text-sm text-gray-600 mb-4 leading-relaxed">
                  {template.description}
                </p>
                <div className="flex flex-wrap gap-2">
                  {template.tags.map((tag, tagIndex) => (
                    <span
                      key={tagIndex}
                      className="text-xs font-medium text-gray-700 bg-gray-100 px-2 py-1 rounded"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default Templates;
