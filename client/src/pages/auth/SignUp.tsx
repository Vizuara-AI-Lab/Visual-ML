import React, { useState, useEffect, useRef } from "react";
import { Eye, EyeOff, Mail, Lock, User } from "lucide-react";
import axiosInstance from "../../lib/axios";
import { useNavigate } from "react-router";

const SignUp: React.FC = () => {
  const navigate = useNavigate();
  const googleButtonRef = useRef<HTMLDivElement>(null);
  const [formData, setFormData] = useState({
    emailId: "",
    password: "",
    fullName: "",
  });
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [validationErrors, setValidationErrors] = useState<
    Record<string, string>
  >({});

  const validatePassword = (password: string): string[] => {
    const errors: string[] = [];
    if (password.length < 8) errors.push("At least 8 characters");
    if (!/[A-Z]/.test(password)) errors.push("One uppercase letter");
    if (!/[a-z]/.test(password)) errors.push("One lowercase letter");
    if (!/\d/.test(password)) errors.push("One digit");
    return errors;
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
    setError("");

    // Clear specific validation error
    if (validationErrors[name]) {
      const newErrors = { ...validationErrors };
      delete newErrors[name];
      setValidationErrors(newErrors);
    }
  };

  const validateForm = (): boolean => {
    const errors: Record<string, string> = {};

    // Email validation
    if (!formData.emailId) {
      errors.emailId = "Email is required";
    } else if (!/\S+@\S+\.\S+/.test(formData.emailId)) {
      errors.emailId = "Invalid email address";
    }

    // Password validation
    const passwordErrors = validatePassword(formData.password);
    if (passwordErrors.length > 0) {
      errors.password = `Password must have: ${passwordErrors.join(", ")}`;
    }

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setLoading(true);
    setError("");

    try {
      await axiosInstance.post("/auth/student/register", formData);

      // Navigate to OTP verification
      navigate("/verify-email", { state: { email: formData.emailId } });
    } catch (err: unknown) {
      const error = err as {
        response?: { data?: { detail?: string | Array<{ msg: string }> } };
      };
      if (error.response?.data?.detail) {
        if (Array.isArray(error.response.data.detail)) {
          setError(error.response.data.detail.map((e) => e.msg).join(", "));
        } else {
          setError(error.response.data.detail);
        }
      } else {
        setError("Registration failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

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

      console.log("✅ Initializing Google Sign-Up button...");

      // @ts-expect-error - Google Sign-In loaded from CDN
      window.google.accounts.id.initialize({
        client_id: clientId,
        use_fedcm_for_prompt: false,
        callback: handleGoogleCallback,
      });

      // @ts-expect-error - Google Sign-In loaded from CDN
      window.google.accounts.id.renderButton(googleButtonRef.current, {
        theme: "outline",
        size: "large",
        width: googleButtonRef.current.offsetWidth,
        text: "signup_with",
      });
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
    setLoading(true);
    setError("");

    try {
      const result = await axiosInstance.post("/auth/student/google", {
        idToken: response.credential,
      });

      localStorage.setItem("user", JSON.stringify(result.data.user));
      navigate("/dashboard");
    } catch (err: any) {
      setError(
        err.response?.data?.detail ||
          err.response?.data?.message ||
          "Google Sign-Up failed. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  };

  const getPasswordStrength = () => {
    const errors = validatePassword(formData.password);
    if (!formData.password) return { text: "", color: "" };
    if (errors.length === 0) return { text: "Strong", color: "text-green-600" };
    if (errors.length <= 2) return { text: "Medium", color: "text-yellow-600" };
    return { text: "Weak", color: "text-red-600" };
  };

  const passwordStrength = getPasswordStrength();

  return (
    <div className="min-h-screen flex items-center justify-center bg-linear-to-br from-gray-50 to-gray-100 px-4 py-12">
      <div className="max-w-md w-full space-y-8">
        {/* Logo and Header */}
        <div className="text-center">
          <h2 className="mt-6 text-3xl font-bold text-gray-900">
            Create your account
          </h2>
          <p className="mt-2 text-sm text-gray-600">
            Start building ML workflows visually
          </p>
        </div>

        {/* Sign Up Form */}
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
                Name <span className="text-red-500">*</span>
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <User className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  id="fullName"
                  name="fullName"
                  type="text"
                  required
                  value={formData.fullName}
                  onChange={handleChange}
                  className={`block w-full pl-10 pr-3 py-3 border-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#29ABE2] focus:border-transparent transition-all ${
                    validationErrors.fullName
                      ? "border-red-300"
                      : "border-gray-200"
                  }`}
                  placeholder="John Doe"
                />
              </div>
              {validationErrors.fullName && (
                <p className="mt-1 text-sm text-red-600">
                  {validationErrors.fullName}
                </p>
              )}
            </div>
            <div>
              <label
                htmlFor="emailId"
                className="block text-sm font-medium text-gray-700 mb-2"
              >
                Email address <span className="text-red-500">*</span>
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
                  className={`block w-full pl-10 pr-3 py-3 border-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#29ABE2] focus:border-transparent transition-all ${
                    validationErrors.emailId
                      ? "border-red-300"
                      : "border-gray-200"
                  }`}
                  placeholder="you@example.com"
                />
              </div>
              {validationErrors.emailId && (
                <p className="mt-1 text-sm text-red-600">
                  {validationErrors.emailId}
                </p>
              )}
            </div>

            {/* Password Input */}
            <div>
              <label
                htmlFor="password"
                className="block text-sm font-medium text-gray-700 mb-2"
              >
                Password <span className="text-red-500">*</span>
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
                  className={`block w-full pl-10 pr-12 py-3 border-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#29ABE2] focus:border-transparent transition-all ${
                    validationErrors.password
                      ? "border-red-300"
                      : "border-gray-200"
                  }`}
                  placeholder="Create a strong password"
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
              {formData.password && (
                <p
                  className={`mt-1 text-sm font-medium ${passwordStrength.color}`}
                >
                  {passwordStrength.text}
                </p>
              )}
              {validationErrors.password && (
                <p className="mt-1 text-sm text-red-600">
                  {validationErrors.password}
                </p>
              )}
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
                  Creating account...
                </div>
              ) : (
                "Create account"
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

          {/* Google Sign Up Button */}
          <div ref={googleButtonRef} className="w-full"></div>

          {/* Sign In Link */}
          <p className="text-center text-sm text-gray-600">
            Already have an account?{" "}
            <a
              onClick={()=>

                navigate("/signin")
              }
              className="font-medium text-[#29ABE2] hover:text-[#FF00FF] transition-colors hover:underline cursor-pointer"
            >
              Sign in
            </a>
          </p>

          {/* Terms */}
          <p className="text-xs text-center text-gray-500">
            By signing up, you agree to our{" "}
            <a href="#" className="text-[#29ABE2] hover:underline">
              Terms of Service
            </a>{" "}
            and{" "}
            <a href="#" className="text-[#29ABE2] hover:underline">
              Privacy Policy
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default SignUp;
