import React from "react";
import { motion } from "framer-motion";
import { Quote } from "lucide-react";

interface Testimonial {
  quote: string;
  name: string;
  role: string;
  initials: string;
}

const testimonials: Testimonial[] = [
  {
    quote:
      "Visual ML transformed how we teach machine learning. Students can now focus on concepts instead of debugging code. The drag-and-drop interface is incredibly intuitive.",
    name: "Dr. Priya Sharma",
    role: "Professor, Computer Science",
    initials: "PS",
  },
  {
    quote:
      "As a data science student, this tool helped me understand ML pipelines visually. I built my first end-to-end model in minutes instead of hours. Absolutely game-changing.",
    name: "Alex Chen",
    role: "Graduate Student, Data Science",
    initials: "AC",
  },
  {
    quote:
      "The GenAI integration is brilliant. It suggests optimal configurations and helps beginners avoid common pitfalls. Our team's productivity increased significantly.",
    name: "Rahul Verma",
    role: "ML Engineer, Startup Founder",
    initials: "RV",
  },
];

const TestimonialsSection: React.FC = () => {
  return (
    <section className="relative py-32 bg-white overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#f0f0f0_1px,transparent_1px),linear-gradient(to_bottom,#f0f0f0_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-30" />

      <div className="relative max-w-7xl mx-auto px-6 lg:px-8">
        {/* Header */}
        <motion.div
          className="text-center space-y-4 mb-16"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-4xl lg:text-5xl font-bold bg-linear-to-br from-slate-900 via-slate-800 to-slate-900 bg-clip-text text-transparent">
            Loved by learners and builders
          </h2>
          <p className="text-xl text-slate-600 max-w-2xl mx-auto font-light">
            See what our community has to say about Visual ML
          </p>
        </motion.div>

        {/* Testimonial Cards */}
        <div className="grid md:grid-cols-3 gap-6">
          {testimonials.map((testimonial, index) => (
            <motion.div
              key={testimonial.name}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.15 }}
              className="group relative bg-white/80 backdrop-blur-xl border border-slate-200/60 rounded-2xl p-8 hover:shadow-xl hover:shadow-slate-900/10 transition-all duration-300 ring-1 ring-slate-900/5"
            >
              {/* Quote icon */}
              <div className="mb-6">
                <div className="w-10 h-10 rounded-xl bg-linear-to-br from-slate-900 to-slate-700 flex items-center justify-center shadow-lg shadow-slate-900/25">
                  <Quote className="w-5 h-5 text-white" />
                </div>
              </div>

              {/* Quote text */}
              <p className="text-slate-600 leading-relaxed mb-8 text-sm">
                "{testimonial.quote}"
              </p>

              {/* Author */}
              <div className="flex items-center gap-3 pt-6 border-t border-slate-200/60">
                <div className="w-10 h-10 rounded-full bg-linear-to-br from-slate-900 to-slate-700 flex items-center justify-center shadow-md">
                  <span className="text-white text-sm font-semibold">
                    {testimonial.initials}
                  </span>
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-900">
                    {testimonial.name}
                  </p>
                  <p className="text-xs text-slate-500">{testimonial.role}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default TestimonialsSection;
