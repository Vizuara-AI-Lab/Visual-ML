import React, { useState, useEffect } from "react";
import {useNavigate} from "react-router";



const Navbar: React.FC = () => {
  const navigate = useNavigate();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? "bg-white/80 backdrop-blur-lg border-b border-gray-200 shadow-sm"
          : "bg-white border-b border-transparent"
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex-shrink-0">
            <span className="text-xl font-semibold text-gray-900">
              Visual ML
            </span>
          </div>

          <div className="hidden md:flex items-center space-x-8">
            <button
              onClick={() => scrollToSection("features")}
              className="text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors"
            >
              Features
            </button>
            <button
              onClick={() => scrollToSection("genai")}
              className="text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors"
            >
              GenAI
            </button>
            <button
              onClick={() => scrollToSection("how-it-works")}
              className="text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors"
            >
              How It Works
            </button>
            <button
              onClick={() => scrollToSection("templates")}
              className="text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors"
            >
              Templates
            </button>
            <button
              onClick={() => scrollToSection("testimonials")}
              className="text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors"
            >
              Testimonials
            </button>
            <button
              onClick={() => scrollToSection("faq")}
              className="text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors"
            >
              FAQ
            </button>
          </div>

          <div className="flex items-center space-x-4">
            <button onClick={() => navigate("/signin")} className="text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors px-4 py-2">
            
              Sign in
            </button>
            <button onClick={() => navigate("/signup")} className="text-sm font-medium text-white bg-gray-900 hover:bg-gray-800 transition-colors px-5 py-2 rounded-lg">
              Get started
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
