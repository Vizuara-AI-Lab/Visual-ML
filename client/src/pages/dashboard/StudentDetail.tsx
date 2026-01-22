import React from "react";
import { useNavigate, useParams } from "react-router";
import { useStudentDetail } from "../../hooks/queries/useStudentDetail";
import { useUpdateStudent } from "../../hooks/mutations/useUpdateStudent";

const StudentDetail: React.FC = () => {
  const navigate = useNavigate();
  const { id } = useParams();
  
  // Use TanStack Query hooks
  const { data: student, isLoading, error } = useStudentDetail(id);
  const updateStudentMutation = useUpdateStudent();

  const togglePremium = () => {
    if (!student || !id) return;
    updateStudentMutation.mutate({
      id,
      updates: { isPremium: !student.isPremium },
    });
  };

  const toggleActive = () => {
    if (!student || !id) return;
    updateStudentMutation.mutate({
      id,
      updates: { isActive: !student.isActive },
    });
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading student details...</p>
        </div>
      </div>
    );
  }

  if (error || !student) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-white p-8 rounded-lg shadow-md max-w-md w-full">
          <div className="text-center">
            <div className="text-red-500 text-5xl mb-4">⚠️</div>
            <h2 className="text-2xl font-bold text-gray-800 mb-2">Error</h2>
            <p className="text-gray-600 mb-6">
              {error instanceof Error ? error.message : "Student not found"}
            </p>
            <button
              onClick={() => navigate("/admin/dashboard")}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Back to Dashboard
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <div className="bg-white border-b shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => navigate("/admin/dashboard")}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <svg
                  className="w-6 h-6 text-gray-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M15 19l-7-7 7-7"
                  />
                </svg>
              </button>
              <h1 className="text-2xl font-bold text-gray-800">
                Student Details
              </h1>
            </div>
            <button
              onClick={() => navigate("/admin/dashboard")}
              className="px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors"
            >
              Back to Dashboard
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Profile Card */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-md overflow-hidden">
              <div className="bg-gradient-to-r from-blue-600 to-purple-600 h-32"></div>
              <div className="px-6 pb-6">
                <div className="flex flex-col items-center -mt-16">
                  <div className="w-32 h-32 rounded-full border-4 border-white bg-white shadow-lg flex items-center justify-center overflow-hidden">
                    {student.profilePic ? (
                      <img
                        src={student.profilePic}
                        alt={student.emailId}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full bg-gradient-to-br from-blue-400 to-purple-500 flex items-center justify-center">
                        <span className="text-4xl font-bold text-white">
                          {student.emailId.charAt(0).toUpperCase()}
                        </span>
                      </div>
                    )}
                  </div>
                  <h2 className="mt-4 text-xl font-bold text-gray-800">
                    {student.emailId}
                  </h2>
                  <p className="text-sm text-gray-500 mt-1">
                    Student ID: #{student.id}
                  </p>

                  {/* Status Badges */}
                  <div className="flex gap-2 mt-4">
                    <span
                      className={`px-3 py-1 rounded-full text-xs font-semibold ${
                        student.isPremium
                          ? "bg-yellow-100 text-yellow-800"
                          : "bg-gray-100 text-gray-800"
                      }`}
                    >
                      {student.isPremium ? "Premium" : "Free"}
                    </span>
                    <span
                      className={`px-3 py-1 rounded-full text-xs font-semibold ${
                        student.isActive
                          ? "bg-green-100 text-green-800"
                          : "bg-red-100 text-red-800"
                      }`}
                    >
                      {student.isActive ? "Active" : "Inactive"}
                    </span>
                  </div>

                  {/* Action Buttons */}
                  <div className="w-full mt-6 space-y-3">
                    <button
                      onClick={togglePremium}
                      disabled={updateStudentMutation.isPending}
                      className="w-full py-2 px-4 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {student.isPremium ? "Revoke Premium" : "Grant Premium"}
                    </button>
                    <button
                      onClick={toggleActive}
                      disabled={updateStudentMutation.isPending}
                      className={`w-full py-2 px-4 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                        student.isActive
                          ? "bg-red-600 hover:bg-red-700"
                          : "bg-green-600 hover:bg-green-700"
                      }`}
                    >
                      {student.isActive
                        ? "Deactivate Account"
                        : "Activate Account"}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Details Card */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-md p-6">
              <h3 className="text-xl font-bold text-gray-800 mb-6">
                Student Information
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Email */}
                <div>
                  <label className="block text-sm font-medium text-gray-500 mb-1">
                    Email Address
                  </label>
                  <p className="text-gray-800 font-medium">{student.emailId}</p>
                </div>

                {/* Auth Provider */}
                <div>
                  <label className="block text-sm font-medium text-gray-500 mb-1">
                    Authentication Method
                  </label>
                  <div className="flex items-center gap-2">
                    <span
                      className={`px-3 py-1 rounded-full text-xs font-semibold ${
                        student.authProvider === "GOOGLE"
                          ? "bg-blue-100 text-blue-800"
                          : "bg-gray-100 text-gray-800"
                      }`}
                    >
                      {student.authProvider}
                    </span>
                  </div>
                </div>

                {/* College/School */}
                <div>
                  <label className="block text-sm font-medium text-gray-500 mb-1">
                    College/School
                  </label>
                  <p className="text-gray-800 font-medium">
                    {student.collegeOrSchool || "Not provided"}
                  </p>
                </div>

                {/* Contact */}
                <div>
                  <label className="block text-sm font-medium text-gray-500 mb-1">
                    Contact Number
                  </label>
                  <p className="text-gray-800 font-medium">
                    {student.contactNo || "Not provided"}
                  </p>
                </div>

                {/* Created At */}
                <div>
                  <label className="block text-sm font-medium text-gray-500 mb-1">
                    Account Created
                  </label>
                  <p className="text-gray-800 font-medium">
                    {new Date(student.createdAt).toLocaleDateString("en-US", {
                      year: "numeric",
                      month: "long",
                      day: "numeric",
                    })}
                  </p>
                </div>

                {/* Last Login */}
                <div>
                  <label className="block text-sm font-medium text-gray-500 mb-1">
                    Last Login
                  </label>
                  <p className="text-gray-800 font-medium">
                    {student.lastLogin
                      ? new Date(student.lastLogin).toLocaleDateString(
                          "en-US",
                          {
                            year: "numeric",
                            month: "long",
                            day: "numeric",
                          },
                        )
                      : "Never"}
                  </p>
                </div>
              </div>

              {/* Recent Project */}
              {student.recentProject && (
                <div className="mt-6 pt-6 border-t">
                  <label className="block text-sm font-medium text-gray-500 mb-2">
                    Recent Project
                  </label>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-gray-700">{student.recentProject}</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StudentDetail;
