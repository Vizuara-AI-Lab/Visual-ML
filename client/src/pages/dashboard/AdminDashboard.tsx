import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router";
import { motion } from "framer-motion";
import {
  Users,
  Crown,
  Activity,
  Search,
  Filter,
  Eye,
  UserCheck,
  UserX,
  Sparkles,
  LogOut,
  TrendingUp,
} from "lucide-react";
import { useAdminProfile } from "../../hooks/queries/useAdminProfile";
import { useStudentsList } from "../../hooks/queries/useStudentsList";
import { useUpdateStudent } from "../../hooks/mutations/useUpdateStudent";

const AdminDashboard: React.FC = () => {
  const navigate = useNavigate();
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [filter, setFilter] = useState({ isPremium: "", isActive: "" });

  // Debounce search input (500ms delay)
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(search);
    }, 500);

    return () => clearTimeout(timer);
  }, [search]);

  // Use TanStack Query hooks
  const { data: adminProfile } = useAdminProfile();
  const {
    data: students = [],
    isLoading,
    isFetching,
    error,
  } = useStudentsList({
    search: debouncedSearch || undefined,
    isPremium: filter.isPremium ? filter.isPremium === "true" : undefined,
    isActive: filter.isActive ? filter.isActive === "true" : undefined,
  });
  const updateStudentMutation = useUpdateStudent();

  const handleLogout = () => {
    localStorage.clear();
    navigate("/admin/login");
  };

  const togglePremium = async (studentId: number, currentStatus: boolean) => {
    updateStudentMutation.mutate({
      id: studentId,
      updates: { isPremium: !currentStatus },
    });
  };

  const toggleActive = async (studentId: number, currentStatus: boolean) => {
    updateStudentMutation.mutate({
      id: studentId,
      updates: { isActive: !currentStatus },
    });
  };

  const totalStudents = students?.length || 0;
  const premiumStudents = students?.filter((s) => s.isPremium).length || 0;
  const activeStudents = students?.filter((s) => s.isActive).length || 0;

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50 flex items-center justify-center">
        <div className="text-center">
          <div className="relative">
            <div className="animate-spin rounded-full h-16 w-16 border-4 border-slate-200 border-t-slate-900 mx-auto"></div>
            <div className="absolute inset-0 rounded-full bg-gradient-to-tr from-slate-900/10 to-transparent blur-xl"></div>
          </div>
          <p className="text-slate-600 mt-4 font-medium">
            Loading dashboard...
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50 flex items-center justify-center">
        <div className="bg-white/80 backdrop-blur-xl p-8 rounded-2xl shadow-xl shadow-slate-900/10 border border-slate-200/60 max-w-md w-full ring-1 ring-slate-900/5">
          <div className="text-center">
            <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-red-500 to-red-600 flex items-center justify-center shadow-lg shadow-red-500/25">
              <span className="text-white text-3xl">⚠️</span>
            </div>
            <h2 className="text-2xl font-bold bg-linear-to-r from-slate-900 to-slate-700 bg-clip-text text-transparent mb-2">
              Error
            </h2>
            <p className="text-slate-600 mb-6">
              {error instanceof Error ? error.message : "Failed to load data"}
            </p>
            <button
              onClick={() => window.location.reload()}
              className="px-6 py-3 bg-slate-900 text-white rounded-xl hover:bg-slate-800 transition-all shadow-lg shadow-slate-900/25 hover:shadow-xl hover:shadow-slate-900/40 font-medium"
            >
              Retry
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
          <div className="flex justify-between items-center">
            <div>
              <div className="flex items-center gap-3 mb-1">
                <div className="relative w-10 h-10 rounded-lg bg-gradient-to-br from-slate-900 to-slate-700 flex items-center justify-center shadow-md shadow-slate-900/25">
                  <Sparkles className="w-5 h-5 text-white" />
                  <div className="absolute inset-0 rounded-lg bg-gradient-to-br from-white/20 to-transparent" />
                </div>
                <h1 className="text-2xl font-bold bg-linear-to-r from-slate-900 to-slate-700 bg-clip-text text-transparent">
                  Admin Dashboard
                </h1>
              </div>
              {adminProfile && (
                <p className="text-sm text-slate-600 ml-13">
                  Welcome back,{" "}
                  <span className="font-medium text-slate-900">
                    {adminProfile.name || adminProfile.email}
                  </span>
                </p>
              )}
            </div>
            <div className="flex items-center gap-4">
              {adminProfile && (
                <div className="hidden sm:block text-right">
                  <p className="text-sm font-semibold text-slate-900">
                    {adminProfile.name || "Admin"}
                  </p>
                  <p className="text-xs text-slate-500">{adminProfile.email}</p>
                </div>
              )}
              <button
                onClick={handleLogout}
                className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-xl hover:bg-red-700 transition-all shadow-lg shadow-red-500/25 hover:shadow-xl hover:shadow-red-500/40 border border-red-700/20 font-medium"
              >
                <LogOut className="w-4 h-4" />
                <span className="hidden sm:inline">Logout</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="group bg-white/80 backdrop-blur-xl p-6 rounded-2xl border border-slate-200/60 shadow-lg shadow-slate-900/5 hover:shadow-xl hover:shadow-slate-900/10 transition-all duration-300 ring-1 ring-slate-900/5"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-500 uppercase tracking-wide">
                  Total Students
                </p>
                <p className="text-4xl font-bold bg-gradient-to-br from-slate-900 to-slate-700 bg-clip-text text-transparent mt-2">
                  {totalStudents}
                </p>
                <div className="flex items-center gap-1.5 mt-2">
                  <TrendingUp className="w-3.5 h-3.5 text-blue-600" />
                  <span className="text-xs text-blue-600 font-semibold">
                    All users
                  </span>
                </div>
              </div>
              <div className="relative w-14 h-14 rounded-xl bg-gradient-to-br from-slate-900 to-slate-700 flex items-center justify-center shadow-lg shadow-slate-900/25 group-hover:scale-110 transition-transform">
                <Users className="h-7 w-7 text-white" />
                <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-white/20 to-transparent" />
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="group bg-white/80 backdrop-blur-xl p-6 rounded-2xl border border-slate-200/60 shadow-lg shadow-slate-900/5 hover:shadow-xl hover:shadow-amber-500/10 transition-all duration-300 ring-1 ring-slate-900/5"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-500 uppercase tracking-wide">
                  Premium Users
                </p>
                <p className="text-4xl font-bold bg-gradient-to-br from-slate-900 to-slate-700 bg-clip-text text-transparent mt-2">
                  {premiumStudents}
                </p>
                <div className="flex items-center gap-1.5 mt-2">
                  <span className="text-xs text-slate-500">
                    {totalStudents > 0
                      ? Math.round((premiumStudents / totalStudents) * 100)
                      : 0}
                    % of total
                  </span>
                </div>
              </div>
              <div className="relative w-14 h-14 rounded-xl bg-gradient-to-br from-amber-500 to-amber-600 flex items-center justify-center shadow-lg shadow-amber-500/25 group-hover:scale-110 transition-transform">
                <Crown className="h-7 w-7 text-white" />
                <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-white/20 to-transparent" />
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="group bg-white/80 backdrop-blur-xl p-6 rounded-2xl border border-slate-200/60 shadow-lg shadow-slate-900/5 hover:shadow-xl hover:shadow-emerald-500/10 transition-all duration-300 ring-1 ring-slate-900/5"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-500 uppercase tracking-wide">
                  Active Users
                </p>
                <p className="text-4xl font-bold bg-gradient-to-br from-slate-900 to-slate-700 bg-clip-text text-transparent mt-2">
                  {activeStudents}
                </p>
                <div className="flex items-center gap-1.5 mt-2">
                  <span className="text-xs text-emerald-600 font-semibold">
                    {totalStudents > 0
                      ? Math.round((activeStudents / totalStudents) * 100)
                      : 0}
                    % active
                  </span>
                </div>
              </div>
              <div className="relative w-14 h-14 rounded-xl bg-gradient-to-br from-emerald-500 to-emerald-600 flex items-center justify-center shadow-lg shadow-emerald-500/25 group-hover:scale-110 transition-transform">
                <Activity className="h-7 w-7 text-white" />
                <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-white/20 to-transparent" />
              </div>
            </div>
          </motion.div>
        </div>
        {/* Filters Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="bg-white/80 backdrop-blur-xl p-6 rounded-2xl border border-slate-200/60 shadow-lg shadow-slate-900/5 mb-6 ring-1 ring-slate-900/5"
        >
          <div className="flex items-center gap-2 mb-4">
            <Filter className="w-5 h-5 text-slate-700" />
            <h3 className="text-lg font-bold text-slate-900">
              Filters & Search
            </h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="relative md:col-span-2">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
              <input
                type="text"
                placeholder="Search by name, email, or college..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full pl-12 pr-10 py-3 bg-white/60 border border-slate-200/60 rounded-xl focus:outline-none focus:ring-2 focus:ring-slate-900 focus:border-transparent transition-all shadow-sm hover:shadow-md backdrop-blur-sm"
              />
              {search !== debouncedSearch && (
                <div className="absolute right-4 top-1/2 -translate-y-1/2">
                  <div className="animate-spin h-5 w-5 border-2 border-slate-900 border-t-transparent rounded-full"></div>
                </div>
              )}
            </div>

            <select
              value={filter.isPremium}
              onChange={(e) =>
                setFilter({ ...filter, isPremium: e.target.value })
              }
              className="px-4 py-3 bg-white/60 border border-slate-200/60 rounded-xl focus:outline-none focus:ring-2 focus:ring-slate-900 focus:border-transparent transition-all shadow-sm hover:shadow-md backdrop-blur-sm font-medium text-slate-700"
            >
              <option value="">All Students</option>
              <option value="true">Premium Only</option>
              <option value="false">Free Only</option>
            </select>

            <select
              value={filter.isActive}
              onChange={(e) =>
                setFilter({ ...filter, isActive: e.target.value })
              }
              className="px-4 py-3 bg-white/60 border border-slate-200/60 rounded-xl focus:outline-none focus:ring-2 focus:ring-slate-900 focus:border-transparent transition-all shadow-sm hover:shadow-md backdrop-blur-sm font-medium text-slate-700"
            >
              <option value="">All Status</option>
              <option value="true">Active Only</option>
              <option value="false">Inactive Only</option>
            </select>
          </div>
        </motion.div>

        {/* Students Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.5 }}
          className="bg-white/80 backdrop-blur-xl rounded-2xl border border-slate-200/60 shadow-xl shadow-slate-900/5 ring-1 ring-slate-900/5 overflow-hidden"
        >
          <div className="p-6 border-b border-slate-200/60 flex items-center justify-between">
            <h2 className="text-2xl font-bold bg-linear-to-r from-slate-900 to-slate-700 bg-clip-text text-transparent">
              Students ({students.length})
            </h2>
            {isFetching && !isLoading && (
              <div className="flex items-center gap-2 text-slate-500">
                <div className="animate-spin h-4 w-4 border-2 border-slate-400 border-t-slate-900 rounded-full"></div>
                <span className="text-xs font-medium">Updating...</span>
              </div>
            )}
          </div>

          <div className="overflow-x-auto relative">
            {isFetching && !isLoading && (
              <div className="absolute inset-0 bg-white/60 backdrop-blur-[1px] z-10" />
            )}
            <table className="w-full">
              <thead className="bg-linear-to-r from-slate-50 to-slate-100/50 border-b border-slate-200/60">
                <tr>
                  <th className="p-4 text-left text-xs font-semibold text-slate-700 uppercase tracking-wide">
                    ID
                  </th>
                  <th className="p-4 text-left text-xs font-semibold text-slate-700 uppercase tracking-wide">
                    Name
                  </th>
                  <th className="p-4 text-left text-xs font-semibold text-slate-700 uppercase tracking-wide">
                    Email
                  </th>
                  <th className="p-4 text-left text-xs font-semibold text-slate-700 uppercase tracking-wide">
                    College/School
                  </th>
                  <th className="p-4 text-left text-xs font-semibold text-slate-700 uppercase tracking-wide">
                    Premium
                  </th>
                  <th className="p-4 text-left text-xs font-semibold text-slate-700 uppercase tracking-wide">
                    Status
                  </th>
                  <th className="p-4 text-left text-xs font-semibold text-slate-700 uppercase tracking-wide">
                    Joined
                  </th>
                  <th className="p-4 text-left text-xs font-semibold text-slate-700 uppercase tracking-wide">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200/60">
                {students.map((student, index) => (
                  <motion.tr
                    key={student.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                    className="hover:bg-slate-50/50 transition-colors"
                  >
                    <td className="p-4 text-sm text-slate-900 font-medium">
                      #{student.id}
                    </td>
                    <td className="p-4">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-slate-900 to-slate-700 flex items-center justify-center text-white text-xs font-semibold shadow-md">
                          {student.fullName?.[0]?.toUpperCase() || "U"}
                        </div>
                        <span className="font-semibold text-slate-900">
                          {student.fullName}
                        </span>
                      </div>
                    </td>
                    <td className="p-4 text-sm text-slate-600">
                      {student.emailId}
                    </td>
                    <td className="p-4 text-sm text-slate-600">
                      {student.collegeOrSchool || "-"}
                    </td>
                    <td className="p-4">
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
                    </td>
                    <td className="p-4">
                      <span
                        className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold border ${
                          student.isActive
                            ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                            : "bg-red-50 text-red-700 border-red-200"
                        }`}
                      >
                        {student.isActive ? "Active" : "Inactive"}
                      </span>
                    </td>
                    <td className="p-4 text-sm text-slate-600">
                      {new Date(student.createdAt).toLocaleDateString()}
                    </td>
                    <td className="p-4">
                      <div className="flex gap-2">
                        <button
                          onClick={() =>
                            navigate(`/admin/students/${student.id}`)
                          }
                          className="p-2 text-slate-700 hover:text-slate-900 bg-slate-100 hover:bg-slate-200 rounded-lg transition-all border border-slate-200 hover:shadow-md group"
                          title="View Details"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() =>
                            togglePremium(student.id, student.isPremium)
                          }
                          className="p-2 text-amber-700 hover:text-amber-900 bg-amber-50 hover:bg-amber-100 rounded-lg transition-all border border-amber-200 hover:shadow-md"
                          title={
                            student.isPremium
                              ? "Revoke Premium"
                              : "Grant Premium"
                          }
                        >
                          <Crown className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() =>
                            toggleActive(student.id, student.isActive)
                          }
                          className={`p-2 rounded-lg transition-all border hover:shadow-md ${
                            student.isActive
                              ? "text-red-700 hover:text-red-900 bg-red-50 hover:bg-red-100 border-red-200"
                              : "text-emerald-700 hover:text-emerald-900 bg-emerald-50 hover:bg-emerald-100 border-emerald-200"
                          }`}
                          title={student.isActive ? "Deactivate" : "Activate"}
                        >
                          {student.isActive ? (
                            <UserX className="w-4 h-4" />
                          ) : (
                            <UserCheck className="w-4 h-4" />
                          )}
                        </button>
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>

          {students.length === 0 && (
            <div className="p-16 text-center">
              <div className="relative inline-block mb-6">
                <div className="absolute inset-0 bg-gradient-to-br from-slate-200 to-slate-100 rounded-full blur-2xl opacity-50"></div>
                <div className="relative w-20 h-20 mx-auto bg-gradient-to-br from-slate-100 to-slate-50 rounded-2xl flex items-center justify-center shadow-lg">
                  <Users className="h-10 w-10 text-slate-400" />
                </div>
              </div>
              <h3 className="text-xl font-bold text-slate-900 mb-2">
                No students found
              </h3>
              <p className="text-sm text-slate-600">
                Try adjusting your search or filters
              </p>
            </div>
          )}
        </motion.div>
      </main>
    </div>
  );
};

export default AdminDashboard;
