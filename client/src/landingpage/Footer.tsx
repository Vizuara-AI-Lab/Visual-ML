import React from "react";
import { motion } from "framer-motion";
import { Twitter, Linkedin, Mail } from "lucide-react";

const Footer: React.FC = () => {
  const navigation = {
    product: [
      { name: "Features", href: "#features" },
      { name: "GenAI", href: "#genai" },
      { name: "Templates", href: "#templates" },
      { name: "Pricing", href: "#" },
    ],
    company: [
      { name: "About", href: "#" },
      { name: "Blog", href: "#" },
      { name: "Careers", href: "#" },
      { name: "Contact", href: "#" },
    ],

    legal: [
      { name: "Privacy", href: "#" },
      { name: "Terms", href: "#" },
      { name: "Security", href: "#" },
      { name: "Cookies", href: "#" },
    ],
  };

  const socials = [
    { icon: Twitter, href: "#" },
    { icon: Linkedin, href: "https://www.linkedin.com/company/vizuara" },
    { icon: Mail, href: "mailto:hello@vizuara.com" },
  ];

  return (
    <footer className="bg-gray-50 border-t border-gray-200">
      <motion.div
        className="max-w-7xl mx-auto px-6 lg:px-8 py-16"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 0.6 }}
      >
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-12">
          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-4">
              Product
            </h3>
            <ul className="space-y-3">
              {navigation.product.map((item) => (
                <li key={item.name}>
                  <a
                    href={item.href}
                    className="text-sm text-gray-600 hover:text-gray-900 transition-colors"
                  >
                    {item.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-4">
              Company
            </h3>
            <ul className="space-y-3">
              {navigation.company.map((item) => (
                <li key={item.name}>
                  <a
                    href={item.href}
                    className="text-sm text-gray-600 hover:text-gray-900 transition-colors"
                  >
                    {item.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>
          
          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-4">Legal</h3>
            <ul className="space-y-3">
              {navigation.legal.map((item) => (
                <li key={item.name}>
                  <a
                    href={item.href}
                    className="text-sm text-gray-600 hover:text-gray-900 transition-colors"
                  >
                    {item.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="pt-8 border-t border-gray-200">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="flex items-center gap-6">
              <span className="text-xl font-semibold text-gray-900">
                Visual ML
              </span>
              <span className="text-sm text-gray-600">
                © 2026 Visual ML. All rights reserved.
              </span>
            </div>
            <div className="flex items-center gap-4">
              {socials.map((social, index) => {
                const Icon = social.icon;
                return (
                  <a
                    key={index}
                    href={social.href}
                    className="w-10 h-10 flex items-center justify-center rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors"
                  >
                    <Icon className="w-5 h-5 text-gray-700" />
                  </a>
                );
              })}
            </div>
          </div>
        </div>
      </motion.div>
    </footer>
  );
};

export default Footer;
