import React from "react";
import { motion } from "framer-motion";
import { Clock, Bot, Mail, Sparkles, Zap, Code2 } from "lucide-react";

const GenAISection: React.FC = () => {
  const features = [
    {
      icon: Clock,
      title: "Cron Scheduling",
      description:
        "Automate web scraping with flexible cron-based scheduling.",
    },
    {
      icon: Bot,
      title: "GPT Integration",
      description:
        "Leverage GPT models for intelligent data analysis, and automated insights generation.",
    },
    {
      icon: Mail,
      title: "Email Automation",
      description:
        "Automatically send reports, alerts, and predictions via email. Keep stakeholders informed in real-time.",
    },
    {
      icon: Sparkles,
      title: "Smart Workflows",
      description:
        "AI-powered workflow optimization and node recommendations based on your data and objectives.",
    },
    {
      icon: Zap,
      title: "Webhook Triggers",
      description:
        "Connect to external services with custom webhooks. Trigger pipelines on any event from any platform.",
    },
    {
      icon: Code2,
      title: "API Automation",
      description:
        "RESTful APIs for every pipeline. Integrate ML into your applications with simple HTTP requests.",
    },
  ];

  return (
    <section
      id="genai"
      className="relative py-32 px-6 lg:px-8 bg-white overflow-hidden"
    >
      {/* Background Elements */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#f0f0f0_1px,transparent_1px),linear-gradient(to_bottom,#f0f0f0_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-40" />
      </div>

      <div className="relative max-w-7xl mx-auto">
        <motion.div
          className="text-center space-y-4 mb-16"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
        >
          {/* Badge */}
          <div className="flex justify-center mb-4">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/80 backdrop-blur-xl border border-slate-200/60 rounded-full">
              <Sparkles className="w-4 h-4 text-emerald-600" />
              <span className="text-sm font-medium text-slate-700">
                Automation Suite
              </span>
            </div>
          </div>

          <h2 className="text-4xl lg:text-5xl font-bold bg-linear-to-br from-slate-900 via-slate-800 to-slate-900 bg-clip-text text-transparent">
            Powerful Automation Built-In
          </h2>
          <p className="text-xl text-slate-600 max-w-3xl mx-auto font-light">
            Schedule workflows, integrate with GPT, send automated emails, and
            connect to any platform. Production-ready automation without the
            complexity.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <motion.div
                key={index}
                className="group relative bg-white/80 backdrop-blur-xl border border-slate-200/60 rounded-2xl p-8 hover:bg-white transition-all duration-300 hover:shadow-xl hover:shadow-slate-900/10 ring-1 ring-slate-900/5"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                whileHover={{
                  y: -4,
                  transition: { duration: 0.2 },
                }}
              >
                {/* Step number */}
                <span className="absolute top-4 right-4 text-xs font-bold text-slate-200">
                  {String(index + 1).padStart(2, "0")}
                </span>

                <div className="w-14 h-14 bg-linear-to-br from-slate-900 to-slate-700 rounded-xl flex items-center justify-center mb-6 shadow-lg shadow-slate-900/25 group-hover:shadow-xl group-hover:shadow-slate-900/30 transition-all">
                  <Icon className="w-7 h-7 text-white" />
                </div>
                <h3 className="text-lg font-semibold text-slate-900 mb-3">
                  {feature.title}
                </h3>
                <p className="text-sm text-slate-600 leading-relaxed">
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
