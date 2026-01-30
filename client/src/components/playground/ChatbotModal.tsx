import { motion, AnimatePresence } from "framer-motion";
import { X, Send, AlertTriangle, MessageSquare } from "lucide-react";
import { useState, useEffect, useRef } from "react";
import { usePlaygroundStore } from "../../store/playgroundStore";

interface ChatbotModalProps {
  nodeId: string | null;
  onClose: () => void;
}

interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
}

export const ChatbotModal = ({ nodeId, onClose }: ChatbotModalProps) => {
  const { getNodeById, edges } = usePlaygroundStore();
  const node = nodeId ? getNodeById(nodeId) : null;

  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Check if connected to LLM node
  const isConnectedToLLM = () => {
    if (!nodeId) return false;
    const incomingEdges = edges.filter((edge) => edge.target === nodeId);
    return incomingEdges.some((edge) => {
      const sourceNode = getNodeById(edge.source);
      return sourceNode?.type === "llm_node";
    });
  };

  const hasLLMConnection = isConnectedToLLM();

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Early return AFTER ALL hooks
  if (!nodeId || !node) return null;

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;
    if (!hasLLMConnection) return;

    const userMessage: Message = {
      role: "user",
      content: inputMessage,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputMessage("");
    setIsLoading(true);

    try {
      // Get LLM Provider node config
      const incomingEdges = edges.filter((edge) => edge.target === nodeId);
      const llmEdge = incomingEdges.find((edge) => {
        const sourceNode = getNodeById(edge.source);
        return sourceNode?.type === "llm_node";
      });

      if (!llmEdge) {
        throw new Error("LLM Provider node not found");
      }

      const llmNode = getNodeById(llmEdge.source);
      if (!llmNode) {
        throw new Error("LLM Provider node not found");
      }

      // Call backend API
      const response = await fetch("/api/v1/genai/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: inputMessage,
          sessionId: node.data.config?.sessionId || "default",
          llmConfig: llmNode.data.config,
          chatbotConfig: node.data.config,
          conversationHistory: messages.map((m) => ({
            role: m.role,
            content: m.content,
          })),
        }),
      });

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || "Failed to get response");
      }

      const assistantMessage: Message = {
        role: "assistant",
        content: data.response,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Chat error:", error);
      const errorMessage: Message = {
        role: "assistant",
        content: `Error: ${error instanceof Error ? error.message : "Failed to get response"}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 bg-black/60 backdrop-blur-sm"
          onClick={onClose}
        />

        <motion.div
          initial={{ scale: 0.95, opacity: 0, y: 20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.95, opacity: 0, y: 20 }}
          transition={{ type: "spring", damping: 25, stiffness: 300 }}
          className="relative bg-gray-900 border border-gray-700 rounded-xl shadow-2xl w-full max-w-3xl h-[80vh] flex flex-col overflow-hidden z-10"
        >
          {/* Header */}
          <div
            className="px-6 py-4 border-b border-gray-700 flex items-center justify-between"
            style={{ borderTopColor: "#10B981", borderTopWidth: "3px" }}
          >
            <div className="flex items-center gap-3">
              <div
                className="p-2 rounded-lg"
                style={{ backgroundColor: "#10B98120" }}
              >
                <MessageSquare className="w-5 h-5" style={{ color: "#10B981" }} />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">
                  Chatbot - {node.data.label}
                </h3>
                <p className="text-sm text-gray-400">
                  Session: {node.data.config?.sessionId || "default"}
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-gray-400" />
            </button>
          </div>

          {/* Warning if not connected to LLM */}
          {!hasLLMConnection && (
            <div className="mx-6 mt-4 p-4 bg-amber-900/20 border border-amber-700 rounded-lg flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-semibold text-amber-200">
                  Not Connected to LLM Provider
                </p>
                <p className="text-xs text-amber-300 mt-1">
                  Please connect an LLM Provider node to this Chatbot node to enable chat functionality.
                </p>
              </div>
            </div>
          )}

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center text-gray-500">
                  <MessageSquare className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p className="text-sm">No messages yet</p>
                  <p className="text-xs mt-1">
                    {hasLLMConnection
                      ? "Start a conversation by typing a message below"
                      : "Connect an LLM Provider node first"}
                  </p>
                </div>
              </div>
            ) : (
              messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${
                    message.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[70%] rounded-lg px-4 py-2 ${
                      message.role === "user"
                        ? "bg-blue-600 text-white"
                        : "bg-gray-800 text-gray-200"
                    }`}
                  >
                    <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                    <p className="text-xs opacity-70 mt-1">
                      {message.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-800 rounded-lg px-4 py-2">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" />
                    <div
                      className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
                      style={{ animationDelay: "0.1s" }}
                    />
                    <div
                      className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
                      style={{ animationDelay: "0.2s" }}
                    />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="p-4 border-t border-gray-700">
            <div className="flex gap-2">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={!hasLLMConnection || isLoading}
                placeholder={
                  hasLLMConnection
                    ? "Type your message... (Press Enter to send, Shift+Enter for new line)"
                    : "Connect an LLM Provider node first..."
                }
                className="flex-1 px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed resize-none"
                rows={2}
              />
              <button
                onClick={handleSendMessage}
                disabled={!hasLLMConnection || isLoading || !inputMessage.trim()}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg flex items-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-blue-600"
              >
                <Send className="w-4 h-4" />
                Send
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};
