import { useNavigate } from "react-router";
import { motion } from "framer-motion";
import { Home, ArrowLeft, Sparkles } from "lucide-react";
import Navbar from "../../landingpage/Navbar";

const NotFound = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-linear-to-b from-white to-slate-50">
      <Navbar />

      {/* Background grid pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#f0f0f0_1px,transparent_1px),linear-gradient(to_bottom,#f0f0f0_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_50%,#000_70%,transparent_110%)]" />

      <div className="relative flex items-center justify-center min-h-screen px-4">
        <div className="max-w-2xl w-full text-center">
          {/* 404 Number */}
          <motion.div
            className="mb-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <h1 className="text-[10rem] lg:text-[12rem] font-bold leading-none bg-linear-to-br from-slate-900 via-slate-700 to-slate-400 bg-clip-text text-transparent select-none">
              404
            </h1>
          </motion.div>

          {/* Error Message */}
          <motion.div
            className="mb-10"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <h2 className="text-3xl font-bold text-slate-900 mb-3">
              Page Not Found
            </h2>
            <p className="text-slate-500 text-lg max-w-md mx-auto leading-relaxed">
              The page you're looking for doesn't exist or has been moved.
            </p>
          </motion.div>

          {/* Action Buttons */}
          <motion.div
            className="flex flex-col sm:flex-row gap-4 justify-center items-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <motion.button
              onClick={() => navigate(-1)}
              className="px-6 py-3 text-slate-700 font-medium hover:text-slate-900 transition-all flex items-center gap-2 bg-white/50 backdrop-blur-sm rounded-xl border border-slate-200/60 hover:bg-white/80 hover:border-slate-300/60 shadow-sm"
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
            >
              <ArrowLeft className="w-4 h-4" />
              Go Back
            </motion.button>

            <motion.button
              onClick={() => navigate("/")}
              className="group relative px-6 py-3 bg-slate-900 text-white font-medium rounded-xl hover:bg-slate-800 transition-all flex items-center gap-2 shadow-lg shadow-slate-900/25 hover:shadow-xl hover:shadow-slate-900/40 overflow-hidden"
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
            >
              <span className="relative z-10 flex items-center gap-2">
                <Home className="w-4 h-4" />
                Back to Home
              </span>
              <div className="absolute inset-0 bg-linear-to-r from-slate-800 to-slate-700 opacity-0 group-hover:opacity-100 transition-opacity" />
            </motion.button>
          </motion.div>

          {/* Decorative Element */}
          <motion.div
            className="mt-16 flex justify-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.4 }}
            transition={{ duration: 0.8, delay: 0.5 }}
          >
            <div className="w-16 h-16 rounded-2xl bg-linear-to-br from-slate-900 to-slate-700 flex items-center justify-center shadow-md shadow-slate-900/25">
              <Sparkles className="w-8 h-8 text-white" />
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default NotFound;
