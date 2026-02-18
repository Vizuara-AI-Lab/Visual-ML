import React, { useState } from "react";
import { motion } from "framer-motion";
import { Twitter, Linkedin, Mail, ArrowRight, Sparkles } from "lucide-react";

const Footer: React.FC = () => {
  const [email, setEmail] = useState("");

  const footerLinks = {
    Product: [
      { name: "Features", href: "#features" },
      { name: "Automation", href: "#genai" },
      { name: "Templates", href: "#templates" },
    ],
    Company: [
      { name: "About", href: "https://vizuara.ai/about-us/" },
      { name: "Careers", href: "https://hiring.vizuara.ai/" },
      { name: "Contact", href: "https://vizuara.ai/contact-us" },
    ],
    Resources: [
      { name: "Documentation", href: "#docs" },
      { name: "API Reference", href: "#api" },
      { name: "Community", href: "#community" },
      { name: "Support", href: "#support" },
    ],
  };

  const socials = [
    { icon: Twitter, href: "https://x.com/VizuaraAI", label: "Twitter" },
    {
      icon: Linkedin,
      href: "https://www.linkedin.com/company/vizuara",
      label: "LinkedIn",
    },
    { icon: Mail, href: "mailto:hello@vizuara.com", label: "Email" },
  ];

  return (
    <footer className="relative bg-white border-t border-slate-200/60">
      {/* Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#f0f0f0_1px,transparent_1px),linear-gradient(to_bottom,#f0f0f0_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-40" />
      </div>

      <div className="relative max-w-7xl mx-auto px-6 lg:px-8">
        {/* Main Footer Content */}
        <div className="py-16 lg:py-20">
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-8 lg:gap-12">
            {/* Brand Section */}
            <motion.div
              className="col-span-2 lg:col-span-2"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
            >
              <div className="space-y-6">
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <div className="relative w-8 h-8 rounded-lg bg-linear-to-br from-slate-900 to-slate-700 flex items-center justify-center shadow-md shadow-slate-900/25">
                      <Sparkles className="w-4 h-4 text-white" />
                    </div>
                    <h3 className="text-2xl font-bold bg-linear-to-br from-slate-900 via-slate-800 to-slate-900 bg-clip-text text-transparent">
                      Visual ML
                    </h3>
                  </div>
                  <p className="text-slate-600 leading-relaxed max-w-xs text-sm">
                    Build production-ready machine learning pipelines with an
                    intuitive visual interface. No code required.
                  </p>
                </div>

                {/* Newsletter */}
                <div>
                  <p className="text-sm font-semibold text-slate-900 mb-3">
                    Stay updated
                  </p>
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      setEmail("");
                    }}
                    className="flex gap-2"
                  >
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="your@email.com"
                      className="flex-1 px-4 py-2.5 bg-white border border-slate-200/60 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-slate-900 focus:border-transparent transition-all shadow-sm"
                    />
                    <button
                      type="submit"
                      className="px-4 py-2.5 bg-slate-900 text-white rounded-xl hover:bg-slate-800 transition-all shadow-lg shadow-slate-900/25 hover:shadow-xl"
                    >
                      <ArrowRight className="w-4 h-4" />
                    </button>
                  </form>
                </div>
              </div>
            </motion.div>

            {/* Links Sections */}
            {Object.entries(footerLinks).map(([category, links], catIndex) => (
              <motion.div
                key={category}
                className="col-span-1"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: (catIndex + 1) * 0.1 }}
              >
                <h4 className="text-sm font-semibold text-slate-900 mb-4">
                  {category}
                </h4>
                <ul className="space-y-3">
                  {links.map((link) => (
                    <li key={link.name}>
                      <a
                        href={link.href}
                        className="text-sm text-slate-600 hover:text-slate-900 transition-colors"
                      >
                        {link.name}
                      </a>
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="py-8 border-t border-slate-200/60">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="flex items-center gap-6">
              <span className="text-sm text-slate-600">
                &copy; 2026 Visual ML. All rights reserved.
              </span>
            </div>

            {/* Social Links */}
            <div className="flex items-center gap-2">
              {socials.map((social) => {
                const Icon = social.icon;
                return (
                  <motion.a
                    key={social.label}
                    href={social.href}
                    aria-label={social.label}
                    className="w-10 h-10 flex items-center justify-center rounded-xl bg-slate-50 hover:bg-slate-100 transition-all hover:shadow-md border border-slate-200/60 hover:border-slate-300"
                    whileHover={{ scale: 1.1, y: -2 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Icon className="w-4 h-4 text-slate-700" />
                  </motion.a>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
