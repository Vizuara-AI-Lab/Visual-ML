import React, { useState } from "react";
import { useNavigate } from "react-router";
import axiosInstance from "../../lib/axios";

const ForgotPassword: React.FC = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [resetToken, setResetToken] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const response = await axiosInstance.post(
        "/auth/student/forgot-password",
        {
          emailId: email,
        },
      );
      setSuccess(true);
      setResetToken(response.data.resetToken); // For development only
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to send reset email");
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    return (
      <div className="min-h-screen bg-gray-50 p-4 flex items-center justify-center">
        <div className="max-w-md w-full bg-white p-6 rounded shadow">
          <h1 className="text-2xl font-bold mb-4 text-green-600">
            Reset Link Sent!
          </h1>
          <p className="mb-4">
            If the email exists, a reset link has been sent to{" "}
            <strong>{email}</strong>
          </p>
          {resetToken && (
            <div className="mb-4 p-3 bg-yellow-100 rounded">
              <p className="text-sm font-bold">Development Mode</p>
              <p className="text-sm">Reset Token: {resetToken}</p>
              <button
                onClick={() =>
                  navigate(`/reset-password?token=${resetToken}`)
                }
                className="mt-2 text-blue-600 underline"
              >
                Click here to reset password
              </button>
            </div>
          )}
          <button
            onClick={() => navigate("/signin")}
            className="w-full px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Back to Sign In
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-4 flex items-center justify-center">
      <div className="max-w-md w-full bg-white p-6 rounded shadow">
        <h1 className="text-2xl font-bold mb-6">Forgot Password</h1>

        {error && (
          <div className="mb-4 p-3 bg-red-100 text-red-800 rounded">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block mb-1 font-medium">Email Address</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-3 py-2 border rounded"
              placeholder="Enter your email"
              required
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? "Sending..." : "Send Reset Link"}
          </button>

          <button
            type="button"
            onClick={() => navigate("/signin")}
            className="w-full px-4 py-2 bg-gray-300 rounded hover:bg-gray-400"
          >
            Back to Sign In
          </button>
        </form>
      </div>
    </div>
  );
};

export default ForgotPassword;
