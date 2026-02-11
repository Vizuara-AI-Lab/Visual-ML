import React from "react";
import { useNavigate, useParams } from "react-router";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  Mail,
  Building2,
  Phone,
  Calendar,
  Clock,
  Crown,
  Shield,
  Activity,
  User,
  FolderKanban,
} from "lucide-react";
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
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50 flex items-center justify-center">
        <div className="text-center">
          <div className="relative">
            <div className="animate-spin rounded-full h-16 w-16 border-4 border-slate-200 border-t-slate-900 mx-auto"></div>
            <div className="absolute inset-0 rounded-full bg-gradient-to-tr from-slate-900/10 to-transparent blur-xl"></div>
          </div>
          <p className="text-slate-600 mt-4 font-medium">
            Loading student details...
          </p>
        </div>
      </div>
    );
  }

  if (error || !student) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50 flex items-center justify-center">
        <div className="bg-white/80 backdrop-blur-xl p-8 rounded-2xl shadow-xl shadow-slate-900/10 border border-slate-200/60 max-w-md w-full ring-1 ring-slate-900/5">
          <div className="text-center">
            <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-red-500 to-red-600 flex items-center justify-center shadow-lg shadow-red-500/25">
              <span className="text-white text-3xl">⚠️</span>
            </div>
            <h2 className="text-2xl font-bold bg-gradient-to-r from-slate-900 to-slate-700 bg-clip-text text-transparent mb-2">
              Error
            </h2>
            <p className="text-slate-600 mb-6">
              {error instanceof Error ? error.message : "Student not found"}
            </p>
            <button
              onClick={() => navigate("/admin/dashboard")}
              className="px-6 py-3 bg-slate-900 text-white rounded-xl hover:bg-slate-800 transition-all shadow-lg shadow-slate-900/25 hover:shadow-xl hover:shadow-slate-900/40 font-medium"
            >
              Back to Dashboard
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50">
      {/* Premium Background Pattern */}
      <div className="fixed inset-0 bg-[linear-gradient(to_right,#f0f0f0_1px,transparent_1px),linear-gradient(to_bottom,#f0f0f0_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-30 pointer-events-none" />

      {/* Header */}
      <header className="sticky top-0 z-40 bg-white/80 backdrop-blur-xl border-b border-slate-200/60 shadow-sm shadow-slate-900/5">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => navigate("/admin/dashboard")}
                className="p-2 hover:bg-slate-100 rounded-lg transition-all group"
              >
                <ArrowLeft className="w-5 h-5 text-slate-600 group-hover:text-slate-900 group-hover:-translate-x-1 transition-all" />
              </button>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-slate-900 to-slate-700 bg-clip-text text-transparent">
                Student Details
              </h1>
            </div>
            <button
              onClick={() => navigate("/admin/dashboard")}
              className="px-4 py-2 text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition-all font-medium"
            >
              Back to Dashboard
            </button>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Profile Card */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="lg:col-span-1"
          >
            <div className="bg-white/80 backdrop-blur-xl rounded-2xl shadow-xl shadow-slate-900/10 overflow-hidden border border-slate-200/60 ring-1 ring-slate-900/5">
              <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-700 h-32 relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-white/10 to-transparent"></div>
                <div className="absolute top-4 right-4">
                  {student.isPremium && (
                    <div className="flex items-center gap-1.5 px-3 py-1.5 bg-amber-500/90 backdrop-blur-sm rounded-full border border-amber-200/50 shadow-lg">
                      <Crown className="w-3.5 h-3.5 text-white" />
                      <span className="text-xs font-semibold text-white">
                        Premium
                      </span>
                    </div>
                  )}
                </div>
              </div>
              <div className="px-6 pb-6">
                <div className="flex flex-col items-center -mt-16">
                  <div className="relative">
                    <div className="w-32 h-32 rounded-2xl border-4 border-white bg-white shadow-xl flex items-center justify-center overflow-hidden">
                      {student.profilePic ? (
                        <img
                          src={student.profilePic}
                          alt={student.emailId}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="w-full h-full bg-gradient-to-br from-slate-900 to-slate-700 flex items-center justify-center">
                          <User className="w-16 h-16 text-white" />
                        </div>
                      )}
                    </div>
                    <div className="absolute -bottom-2 -right-2 w-10 h-10 rounded-xl bg-white shadow-lg border-2 border-white flex items-center justify-center">
                      {student.isActive ? (
                        <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-emerald-500 to-emerald-600 flex items-center justify-center">
                          <Activity className="w-4 h-4 text-white" />
                        </div>
                      ) : (
                        <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-slate-300 to-slate-400 flex items-center justify-center">
                          <Activity className="w-4 h-4 text-white" />
                        </div>
                      )}
                    </div>
                  </div>
                  <h2 className="mt-4 text-xl font-bold text-slate-900">
                    {student.emailId}
                  </h2>
                  <p className="text-sm text-slate-500 mt-1 font-medium">
                    Student ID: #{student.id}
                  </p>

                  {/* Status Badges */}
                  <div className="flex gap-2 mt-4">
                    <span
                      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold border ${
                        student.isPremium
                          ? "bg-amber-50 text-amber-700 border-amber-200"
                          : "bg-slate-50 text-slate-700 border-slate-200"
                      }`}
                    >
                      {student.isPremium && <Crown className="w-3 h-3" />}
                      {student.isPremium ? "Premium" : "Free"}
                    </span>
                    <span
                      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold border ${
                        student.isActive
                          ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                          : "bg-red-50 text-red-700 border-red-200"
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
                      className="group w-full py-3 px-4 bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-600 hover:to-amber-700 text-white rounded-xl font-medium transition-all shadow-lg shadow-amber-500/25 hover:shadow-xl hover:shadow-amber-500/40 disabled:opacity-50 disabled:cursor-not-allowed hover:-translate-y-0.5 flex items-center justify-center gap-2"
                    >
                      <Crown className="w-4 h-4" />
                      {student.isPremium ? "Revoke Premium" : "Grant Premium"}
                    </button>
                    <button
                      onClick={toggleActive}
                      disabled={updateStudentMutation.isPending}
                      className={`group w-full py-3 px-4 text-white rounded-xl font-medium transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed hover:-translate-y-0.5 flex items-center justify-center gap-2 ${
                        student.isActive
                          ? "bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 shadow-red-500/25 hover:shadow-xl hover:shadow-red-500/40"
                          : "bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 shadow-emerald-500/25 hover:shadow-xl hover:shadow-emerald-500/40"
                      }`}
                    >
                      <Shield className="w-4 h-4" />
                      {student.isActive
                        ? "Deactivate Account"
                        : "Activate Account"}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Details Card */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="lg:col-span-2"
          >
            <div className="bg-white/80 backdrop-blur-xl rounded-2xl shadow-xl shadow-slate-900/10 p-6 border border-slate-200/60 ring-1 ring-slate-900/5">
              <h3 className="text-2xl font-bold bg-gradient-to-r from-slate-900 to-slate-700 bg-clip-text text-transparent mb-6">
                Student Information
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Email */}
                <div className="p-4 bg-white/60 backdrop-blur-sm rounded-xl border border-slate-200/60 hover:shadow-lg hover:shadow-slate-900/5 transition-all">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center shadow-md shadow-blue-500/25">
                      <Mail className="w-5 h-5 text-white" />
                    </div>
                    <label className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
                      Email Address
                    </label>
                  </div>
                  <p className="text-slate-900 font-semibold text-sm ml-13">
                    {student.emailId}
                  </p>
                </div>

                {/* Auth Provider */}
                <div className="p-4 bg-white/60 backdrop-blur-sm rounded-xl border border-slate-200/60 hover:shadow-lg hover:shadow-slate-900/5 transition-all">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-violet-500 to-violet-600 flex items-center justify-center shadow-md shadow-violet-500/25">
                      <Shield className="w-5 h-5 text-white" />
                    </div>
                    <label className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
                      Authentication
                    </label>
                  </div>
                  <div className="ml-13">
                    <span
                      className={`inline-flex items-center px-3 py-1.5 rounded-full text-xs font-semibold border ${
                        student.authProvider === "GOOGLE"
                          ? "bg-blue-50 text-blue-700 border-blue-200"
                          : "bg-slate-50 text-slate-700 border-slate-200"
                      }`}
                    >
                      {student.authProvider}
                    </span>
                  </div>
                </div>

                {/* College/School */}
                <div className="p-4 bg-white/60 backdrop-blur-sm rounded-xl border border-slate-200/60 hover:shadow-lg hover:shadow-slate-900/5 transition-all">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-emerald-500 to-emerald-600 flex items-center justify-center shadow-md shadow-emerald-500/25">
                      <Building2 className="w-5 h-5 text-white" />
                    </div>
                    <label className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
                      College/School
                    </label>
                  </div>
                  <p className="text-slate-900 font-semibold text-sm ml-13">
                    {student.collegeOrSchool || "Not provided"}
                  </p>
                </div>

                {/* Contact */}
                <div className="p-4 bg-white/60 backdrop-blur-sm rounded-xl border border-slate-200/60 hover:shadow-lg hover:shadow-slate-900/5 transition-all">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-amber-500 to-amber-600 flex items-center justify-center shadow-md shadow-amber-500/25">
                      <Phone className="w-5 h-5 text-white" />
                    </div>
                    <label className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
                      Contact Number
                    </label>
                  </div>
                  <p className="text-slate-900 font-semibold text-sm ml-13">
                    {student.contactNo || "Not provided"}
                  </p>
                </div>

                {/* Created At */}
                <div className="p-4 bg-white/60 backdrop-blur-sm rounded-xl border border-slate-200/60 hover:shadow-lg hover:shadow-slate-900/5 transition-all">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-slate-900 to-slate-700 flex items-center justify-center shadow-md shadow-slate-900/25">
                      <Calendar className="w-5 h-5 text-white" />
                    </div>
                    <label className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
                      Account Created
                    </label>
                  </div>
                  <p className="text-slate-900 font-semibold text-sm ml-13">
                    {new Date(student.createdAt).toLocaleDateString("en-US", {
                      year: "numeric",
                      month: "long",
                      day: "numeric",
                    })}
                  </p>
                </div>

                {/* Last Login */}
                <div className="p-4 bg-white/60 backdrop-blur-sm rounded-xl border border-slate-200/60 hover:shadow-lg hover:shadow-slate-900/5 transition-all">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-indigo-500 to-indigo-600 flex items-center justify-center shadow-md shadow-indigo-500/25">
                      <Clock className="w-5 h-5 text-white" />
                    </div>
                    <label className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
                      Last Login
                    </label>
                  </div>
                  <p className="text-slate-900 font-semibold text-sm ml-13">
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
                <div className="mt-6 pt-6 border-t border-slate-200/60">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 to-purple-600 flex items-center justify-center shadow-md shadow-purple-500/25">
                      <FolderKanban className="w-5 h-5 text-white" />
                    </div>
                    <label className="text-sm font-semibold text-slate-700 uppercase tracking-wide">
                      Recent Project
                    </label>
                  </div>
                  <div className="bg-gradient-to-br from-slate-50 to-slate-100/50 rounded-xl p-5 border border-slate-200/60 ml-13">
                    <p className="text-slate-900 font-medium">
                      {student.recentProject}
                    </p>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  );
};

export default StudentDetail;
