import React from "react";
import Navbar from "./Navbar";
import Hero from "./Hero";
import GenAISection from "./GenAISection";
import Features from "./Features";
import Footer from "./Footer";

const LandingPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-linear-to-b from-white to-slate-50">
      <Navbar />
      <Hero />
      <GenAISection />
      <Features />

      <Footer />
    </div>
  );
};

export default LandingPage;
