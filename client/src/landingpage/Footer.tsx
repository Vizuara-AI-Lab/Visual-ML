import React from "react";
import { Twitter, Linkedin, Mail } from "lucide-react";

const Footer: React.FC = () => {
  const footerLinks = {
    Product: [
      { name: "Features", href: "#features" },
      { name: "Automation", href: "#automation" },
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
            <div className="col-span-2 lg:col-span-2">
              <div className="space-y-6">
                <div className="space-y-3">
                  <h3 className="text-2xl font-bold bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 bg-clip-text text-transparent">
                    Visual ML
                  </h3>
                  <p className="text-slate-600 leading-relaxed max-w-xs">
                    Build production-ready machine learning pipelines with an
                    intuitive visual interface. No code required.
                  </p>
                </div>

                {/* Newsletter */}
                
              </div>
            </div>

            {/* Links Sections */}
            {Object.entries(footerLinks).map(([category, links]) => (
              <div key={category} className="col-span-1">
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
              </div>
            ))}
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="py-8 border-t border-slate-200/60">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="flex items-center gap-6">
              <span className="text-sm text-slate-600">
                © 2026 Visual ML. All rights reserved.
              </span>
            </div>

            {/* Social Links */}
            <div className="flex items-center gap-2">
              {socials.map((social) => {
                const Icon = social.icon;
                return (
                  <a
                    key={social.label}
                    href={social.href}
                    aria-label={social.label}
                    className="w-10 h-10 flex items-center justify-center rounded-xl bg-slate-50 hover:bg-slate-100 transition-all hover:shadow-md border border-slate-200/60 hover:border-slate-300"
                  >
                    <Icon className="w-4 h-4 text-slate-700" />
                  </a>
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
