import React from "react";
import Navbar from "./Navbar";
import Hero from "./Hero";
import GenAISection from "./GenAISection";
import Features from "./Features";
import HowItWorks from "./HowItWorks";
import Templates from "./Templates";
import Testimonials from "./Testimonials";
import Footer from "./Footer";

const LandingPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <Hero />
      <GenAISection />
      <Features />
      <HowItWorks />
      <Templates />
      <Testimonials />
      <Footer />
    </div>
  );
};

export default LandingPage;
