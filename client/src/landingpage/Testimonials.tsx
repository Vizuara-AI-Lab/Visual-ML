import React from "react";
import { motion } from "framer-motion";
import { Star } from "lucide-react";

const Testimonials: React.FC = () => {
  const testimonials = [
    {
      name: "Sarah Chen",
      role: "ML Engineer at DataCorp",
      content:
        "Visual ML transformed how we prototype and deploy models. The GenAI assistance caught errors I would have missed, and the custom UI export saved us weeks of development time.",
      rating: 5,
    },
    {
      name: "Michael Rodriguez",
      role: "Data Scientist at TechFlow",
      content:
        "The visual pipeline builder is incredibly intuitive. I can now collaborate with non-technical stakeholders by simply sharing a link to my workflow. Game changer for our team.",
      rating: 5,
    },
    {
      name: "Emily Watson",
      role: "Head of AI at Innovate Labs",
      content:
        "We reduced our model development cycle from months to weeks. The seamless hardware integration and real-time performance monitoring are exactly what we needed.",
      rating: 5,
    },
  ];

  return (
    <section id="testimonials" className="py-24 px-6 lg:px-8 bg-gray-50">
      <div className="max-w-7xl mx-auto">
        <motion.div
          className="text-center space-y-4 mb-16"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-4xl lg:text-5xl font-bold text-gray-900">
            Trusted by ML teams worldwide
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            See what data scientists and engineers are saying about Visual ML.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-8">
          {testimonials.map((testimonial, index) => (
            <motion.div
              key={index}
              className="bg-white border border-gray-200 rounded-xl p-8 space-y-4"
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, delay: index * 0.2 }}
              whileHover={{ y: -8, scale: 1.02, transition: { duration: 0.3 } }}
            >
              <div className="flex gap-1">
                {Array.from({ length: testimonial.rating }).map((_, i) => (
                  <Star
                    key={i}
                    className="w-5 h-5 fill-gray-900 text-gray-900"
                  />
                ))}
              </div>
              <p className="text-gray-700 leading-relaxed">
                {testimonial.content}
              </p>
              <div className="pt-4 border-t border-gray-100">
                <div className="font-semibold text-gray-900">
                  {testimonial.name}
                </div>
                <div className="text-sm text-gray-600">{testimonial.role}</div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Testimonials;
