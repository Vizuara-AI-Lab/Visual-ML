import React, { useState } from "react";
import { ChevronDown } from "lucide-react";

const FAQ: React.FC = () => {
  const [openIndex, setOpenIndex] = useState<number | null>(0);

  const faqs = [
    {
      question: "Do I need coding experience to use Visual ML?",
      answer:
        "No coding experience is required. Our visual interface allows you to build complex ML pipelines by dragging and dropping nodes. However, advanced users can also write custom code when needed.",
    },
    {
      question: "What machine learning frameworks are supported?",
      answer:
        "Visual ML supports popular frameworks including TensorFlow, PyTorch, scikit-learn, and Hugging Face Transformers. You can also integrate custom models and frameworks.",
    },
    {
      question: "Can I deploy my models to production?",
      answer:
        "Yes. Visual ML provides one-click deployment options for cloud platforms, edge devices, and on-premise infrastructure. You can also export custom UIs and APIs for your models.",
    },
    {
      question: "How does the GenAI assistance work?",
      answer:
        "Our GenAI engine analyzes your pipeline in real-time, suggesting optimal node configurations, identifying potential errors, and recommending best practices based on your data and objectives.",
    },
    {
      question: "Is my data secure?",
      answer:
        "Absolutely. We use enterprise-grade encryption for data in transit and at rest. You maintain full control over your data, and we never use it to train our models without explicit permission.",
    },
    {
      question: "What kind of support is available?",
      answer:
        "We offer comprehensive documentation, video tutorials, community forums, and dedicated support channels. Enterprise customers receive priority support with guaranteed response times.",
    },
  ];

  const toggleFAQ = (index: number) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <section id="faq" className="py-24 px-6 lg:px-8 bg-white">
      <div className="max-w-3xl mx-auto">
        <div className="text-center space-y-4 mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-gray-900">
            Frequently asked questions
          </h2>
          <p className="text-xl text-gray-600">
            Everything you need to know about Visual ML.
          </p>
        </div>

        <div className="space-y-4">
          {faqs.map((faq, index) => (
            <div
              key={index}
              className="border border-gray-200 rounded-xl overflow-hidden"
            >
              <button
                onClick={() => toggleFAQ(index)}
                className="w-full flex items-center justify-between p-6 text-left hover:bg-gray-50 transition-colors"
              >
                <span className="text-lg font-semibold text-gray-900 pr-8">
                  {faq.question}
                </span>
                <ChevronDown
                  className={`w-5 h-5 text-gray-600 flex-shrink-0 transition-transform ${
                    openIndex === index ? "rotate-180" : ""
                  }`}
                />
              </button>
              {openIndex === index && (
                <div className="px-6 pb-6">
                  <p className="text-gray-600 leading-relaxed">{faq.answer}</p>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FAQ;
