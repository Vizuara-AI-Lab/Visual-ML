import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router";
import {
  Mail,
  ArrowLeft,
  CheckCircle2,
  AlertCircle,
  Loader2,
} from "lucide-react";
import { motion } from "framer-motion";
import axiosInstance from "../../lib/axios";

const ForgotPassword: React.FC = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState("");
  const [emailError, setEmailError] = useState("");
  const [countdown, setCountdown] = useState(5);

  // Email validation
  const validateEmail = (email: string): boolean => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const handleEmailChange = (value: string) => {
    setEmail(value);
    setEmailError("");
    setError("");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setEmailError("");

    // Validate email
    if (!email.trim()) {
      setEmailError("Email address is required");
      return;
    }

    if (!validateEmail(email)) {
      setEmailError("Please enter a valid email address");
      return;
    }

    setLoading(true);

    try {
      await axiosInstance.post("/auth/student/forgot-password", {
        emailId: email,
      });
      setSuccess(true);
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail;

      if (err.response?.status === 429) {
        setError("Too many requests. Please try again later.");
      } else if (err.response?.status === 404) {
        // For security, don't reveal if email exists or not
        setSuccess(true);
      } else {
        setError(
          errorMessage || "Failed to send reset email. Please try again.",
        );
      }
    } finally {
      setLoading(false);
    }
  };

  // Countdown timer for redirect
  useEffect(() => {
    if (success && countdown > 0) {
      const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
      return () => clearTimeout(timer);
    } else if (success && countdown === 0) {
      navigate("/signin");
    }
  }, [success, countdown, navigate]);

  if (success) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center p-4">
        {/* Background Pattern */}
        <div className="absolute inset-0">
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#f0f0f0_1px,transparent_1px),linear-gradient(to_bottom,#f0f0f0_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-40" />
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative max-w-md w-full"
        >
          <div className="bg-white rounded-2xl shadow-xl border border-slate-200/60 p-8 lg:p-10">
            {/* Success Icon */}
            <div className="flex justify-center mb-6">
              <div className="w-16 h-16 rounded-full bg-emerald-50 flex items-center justify-center">
                <CheckCircle2 className="w-8 h-8 text-emerald-600" />
              </div>
            </div>

            {/* Success Message */}
            <div className="text-center space-y-4 mb-8">
              <h1 className="text-2xl lg:text-3xl font-bold text-slate-900">
                Check Your Email
              </h1>
              <p className="text-slate-600 leading-relaxed">
                If an account exists for{" "}
                <strong className="text-slate-900">{email}</strong>, you'll
                receive a password reset link shortly.
              </p>
              <p className="text-sm text-slate-500">
                Didn't receive an email? Check your spam folder or try again.
              </p>
            </div>

            {/* Actions */}
            <div className="space-y-3">
              <button
                onClick={() => navigate("/signin")}
                className="w-full px-6 py-3 bg-slate-900 text-white rounded-xl font-medium hover:bg-slate-800 transition-all shadow-lg shadow-slate-900/25 hover:shadow-xl hover:shadow-slate-900/30"
              >
                Back to Sign In
              </button>

              <p className="text-center text-sm text-slate-500">
                Redirecting in {countdown} second{countdown !== 1 ? "s" : ""}...
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center p-4">
      {/* Background Pattern */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#f0f0f0_1px,transparent_1px),linear-gradient(to_bottom,#f0f0f0_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-40" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative max-w-md w-full"
      >
        <div className="bg-white rounded-2xl shadow-xl border border-slate-200/60 p-8 lg:p-10">
          {/* Back Button */}
          <button
            onClick={() => navigate("/signin")}
            className="flex items-center gap-2 text-slate-600 hover:text-slate-900 transition-colors mb-6 group"
          >
            <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
            <span className="text-sm font-medium">Back to Sign In</span>
          </button>

          {/* Header */}
          <div className="mb-8">
            <div className="w-12 h-12 rounded-xl bg-linear-to-br from-slate-900 to-slate-700 flex items-center justify-center mb-4 shadow-lg shadow-slate-900/25">
              <Mail className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-2xl lg:text-3xl font-bold text-slate-900 mb-2">
              Forgot Password?
            </h1>
            <p className="text-slate-600">
              No worries! Enter your email and we'll send you reset
              instructions.
            </p>
          </div>

          {/* Error Alert */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl flex items-start gap-3"
            >
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="text-sm font-medium text-red-900">Error</p>
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </motion.div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label
                htmlFor="email"
                className="block text-sm font-medium text-slate-900 mb-2"
              >
                Email Address
              </label>
              <div className="relative">
                <div className="absolute left-4 top-1/2 -translate-y-1/2">
                  <Mail className="w-5 h-5 text-slate-400" />
                </div>
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => handleEmailChange(e.target.value)}
                  className={`w-full pl-12 pr-4 py-3 bg-white border rounded-xl transition-all focus:outline-none focus:ring-2 ${
                    emailError
                      ? "border-red-300 focus:ring-red-500 focus:border-red-500"
                      : "border-slate-200 focus:ring-slate-900 focus:border-slate-900"
                  }`}
                  placeholder="you@example.com"
                  disabled={loading}
                  autoComplete="email"
                  autoFocus
                />
              </div>
              {emailError && (
                <motion.p
                  initial={{ opacity: 0, y: -5 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-2 text-sm text-red-600 flex items-center gap-1"
                >
                  <AlertCircle className="w-4 h-4" />
                  {emailError}
                </motion.p>
              )}
            </div>

            <button
              type="submit"
              disabled={loading || !email.trim()}
              className="w-full px-6 py-3 bg-slate-900 text-white rounded-xl font-medium hover:bg-slate-800 transition-all shadow-lg shadow-slate-900/25 hover:shadow-xl hover:shadow-slate-900/30 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Sending Reset Link...</span>
                </>
              ) : (
                <span>Send Reset Link</span>
              )}
            </button>
          </form>

          {/* Help Text */}
          <div className="mt-6 pt-6 border-t border-slate-200">
            <p className="text-sm text-slate-600 text-center">
              Remember your password?{" "}
              <button
                onClick={() => navigate("/signin")}
                className="font-medium text-slate-900 hover:underline"
              >
                Sign in
              </button>
            </p>
          </div>
        </div>

        {/* Security Note */}
        <p className="mt-6 text-center text-xs text-slate-500 max-w-sm mx-auto">
          For security reasons, we'll send a reset link even if the email
          doesn't exist in our system.
        </p>
      </motion.div>
    </div>
  );
};

export default ForgotPassword;
