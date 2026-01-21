import React, { useState, useEffect, useRef } from "react";
import { Eye, EyeOff, Mail, Lock } from "lucide-react";
import axiosInstance from "../../lib/axios";
import { useNavigate } from "react-router";

const SignIn: React.FC = () => {
  const navigate = useNavigate();
  const googleButtonRef = useRef<HTMLDivElement>(null);
  const [formData, setFormData] = useState({
    emailId: "",
    password: "",
  });
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    // Initialize Google Sign-In button only once
    const initGoogleSignIn = () => {
      // @ts-expect-error - Google Sign-In loaded from CDN
      if (!window.google || !googleButtonRef.current) {
        console.log("Waiting for Google SDK or button ref...");
        return;
      }

      const clientId = import.meta.env.VITE_GOOGLE_CLIENT_ID;
      if (!clientId) {
        console.error("VITE_GOOGLE_CLIENT_ID not configured");
        return;
      }

      console.log("✅ Initializing Google Sign-In button...");

      // @ts-expect-error - Google Sign-In loaded from CDN
      window.google.accounts.id.initialize({
        client_id: clientId,
        use_fedcm_for_prompt: false,
        callback: handleGoogleCallback,
      });

      // @ts-expect-error - Google Sign-In loaded from CDN
      window.google.accounts.id.renderButton(
        googleButtonRef.current,
        {
          theme: "outline",
          size: "large",
          width: googleButtonRef.current.offsetWidth,
          text: "signin_with",
        }
      );

      console.log("✅ Google Sign-In button rendered");
    };

    // Wait for Google SDK to load
    const checkGoogleLoaded = setInterval(() => {
      // @ts-expect-error - Google Sign-In loaded from CDN
      if (window.google) {
        clearInterval(checkGoogleLoaded);
        initGoogleSignIn();
      }
    }, 100);

    return () => clearInterval(checkGoogleLoaded);
  }, []);

  const handleGoogleCallback = async (response: any) => {
    console.log("🎉 Google Sign-In callback triggered");
    setLoading(true);
    setError("");

    try {
      console.log("📤 Sending ID token to backend...");
      const result = await axiosInstance.post("/auth/student/google", {
        idToken: response.credential,
      });

      console.log("✅ Authentication successful!");
      
      // Store tokens
      localStorage.setItem("accessToken", result.data.tokens.accessToken);
      localStorage.setItem("refreshToken", result.data.tokens.refreshToken);
      localStorage.setItem("user", JSON.stringify(result.data.user));

      // Navigate to dashboard
      navigate("/dashboard");
    } catch (err: any) {
      console.error("❌ Google Sign-In failed:", err);
      setError(
        err.response?.data?.detail ||
        err.response?.data?.message ||
        "Google Sign-In failed. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
    setError("");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const response = await axiosInstance.post(
        "/auth/student/login",
        formData,
      );

      // Store tokens
      localStorage.setItem("accessToken", response.data.tokens.accessToken);
      localStorage.setItem("refreshToken", response.data.tokens.refreshToken);
      localStorage.setItem("user", JSON.stringify(response.data.user));

      // Navigate to dashboard
      navigate("/dashboard");
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } };
      setError(
        error.response?.data?.detail ||
          "Login failed. Please check your credentials.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-linear-to-br from-gray-50 to-gray-100 px-4 py-12">
      <div className="max-w-md w-full space-y-8">
        {/* Logo and Header */}
        <div className="text-center">
          <h2 className="mt-6 text-3xl font-bold text-gray-900">
            Welcome back
          </h2>
          <p className="mt-2 text-sm text-gray-600">
            Sign in to your account to continue
          </p>
        </div>

        {/* Sign In Form */}
        <div className="bg-white rounded-2xl shadow-xl p-8 space-y-6">
          {error && (
            <div className="bg-red-50 border-2 border-red-200 rounded-lg p-4">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Email Input */}
            <div>
              <label
                htmlFor="emailId"
                className="block text-sm font-medium text-gray-700 mb-2"
              >
                Email address
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Mail className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  id="emailId"
                  name="emailId"
                  type="email"
                  required
                  value={formData.emailId}
                  onChange={handleChange}
                  className="block w-full pl-10 pr-3 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#29ABE2] focus:border-transparent transition-all"
                  placeholder="you@example.com"
                />
              </div>
            </div>

            {/* Password Input */}
            <div>
              <label
                htmlFor="password"
                className="block text-sm font-medium text-gray-700 mb-2"
              >
                Password
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  id="password"
                  name="password"
                  type={showPassword ? "text" : "password"}
                  required
                  value={formData.password}
                  onChange={handleChange}
                  className="block w-full pl-10 pr-12 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#29ABE2] focus:border-transparent transition-all"
                  placeholder="Enter your password"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 pr-3 flex items-center"
                >
                  {showPassword ? (
                    <EyeOff className="h-5 w-5 text-gray-400 hover:text-gray-600" />
                  ) : (
                    <Eye className="h-5 w-5 text-gray-400 hover:text-gray-600" />
                  )}
                </button>
              </div>
            </div>

            {/* Forgot Password Link */}
            <div className="flex items-center justify-end">
              <a
                href="/auth/forgot-password"
                className="text-sm font-medium text-[#29ABE2] hover:text-[#FF00FF] transition-colors"
              >
                Forgot your password?
              </a>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-linear-to-r from-[#29ABE2] to-[#FF00FF] hover:shadow-lg hover:shadow-[#29ABE2]/50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#29ABE2] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              {loading ? (
                <div className="flex items-center">
                  <svg
                    className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  Signing in...
                </div>
              ) : (
                "Sign in"
              )}
            </button>
          </form>

          {/* Divider */}
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-300"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-white text-gray-500">
                Or continue with
              </span>
            </div>
          </div>

          {/* Google Sign In Button */}
          <div ref={googleButtonRef} className="w-full"></div>

          {/* Sign Up Link */}
          <p className="text-center text-sm text-gray-600">
            Don't have an account?{" "}
            <a
              onClick={() => navigate("/auth/signup")}
              className="font-medium text-[#29ABE2] hover:text-[#FF00FF] transition-colors cursor-pointer"
            >
              Sign up
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default SignIn;
