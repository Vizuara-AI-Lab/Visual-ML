import React, { useState } from "react";
import { useNavigate } from "react-router";
import {
  LogOut,
  FolderKanban,
  Database,
  PlayCircle,
  Settings,
  Plus,
  Search,
  Clock,
  CheckCircle,
  XCircle,
  Trash2,
  Sparkles,
  TrendingUp,
  Zap,
} from "lucide-react";
import { motion } from "framer-motion";
import { useProjects } from "../../hooks/queries/useProjects";
import { useDeleteProject } from "../../hooks/mutations/useDeleteProject";
import { CreateProjectModal } from "../../components/projects/CreateProjectModal";
import { useAllDatasets } from "../../hooks/queries/useAllDatasets";

interface User {
  emailId: string;
  collegeOrSchool?: string;
  profilePic?: string;
  isPremium: boolean;
}

interface Run {
  id: string;
  projectId: string;
  status: "pending" | "running" | "completed" | "failed";
  createdAt: string;
}

const StudentDashboard: React.FC = () => {
  const navigate = useNavigate();
  const [user, setUser] = useState<User | null>(null);
  const [recentRuns] = useState<Run[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);

  const { data: projects = [], isLoading } = useProjects();
  const deleteProject = useDeleteProject();
  const { data: allDatasetsData } = useAllDatasets();

  const loadUserData = () => {
    const userData = localStorage.getItem("user");
    if (userData) {
      setUser(JSON.parse(userData));
    }
  };

  React.useEffect(() => {
    loadUserData();
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("accessToken");
    localStorage.removeItem("refreshToken");
    localStorage.removeItem("user");
    navigate("/signin");
  };

  const handleCreateProject = (projectId: number) => {
    setIsCreateModalOpen(false);
    navigate(`/playground/${projectId}`);
  };

  const handleProjectClick = (projectId: number) => {
    navigate(`/playground/${projectId}`);
  };

  const handleDeleteProject = async (
    e: React.MouseEvent,
    projectId: number,
  ) => {
    e.stopPropagation();
    if (window.confirm("Are you sure you want to delete this project?")) {
      await deleteProject.mutateAsync(projectId);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-5 w-5 text-green-600" />;
      case "failed":
        return <XCircle className="h-5 w-5 text-red-600" />;
      case "running":
        return <Clock className="h-5 w-5 text-blue-600 animate-spin" />;
      default:
        return <Clock className="h-5 w-5 text-gray-400" />;
    }
  };

  const filteredProjects = projects.filter((project) =>
    project.name.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 via-white to-slate-50">
        <div className="relative">
          <div className="animate-spin rounded-full h-16 w-16 border-4 border-slate-200 border-t-slate-900"></div>
          <div className="absolute inset-0 rounded-full bg-gradient-to-tr from-slate-900/10 to-transparent blur-xl"></div>
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
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <div className="flex items-center gap-2">
                <div className="relative w-8 h-8 rounded-lg bg-gradient-to-br from-slate-900 to-slate-700 flex items-center justify-center shadow-md shadow-slate-900/25">
                  <Sparkles className="w-4 h-4 text-white" />
                  <div className="absolute inset-0 rounded-lg bg-gradient-to-br from-white/20 to-transparent" />
                </div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-slate-900 to-slate-700 bg-clip-text text-transparent">
                  Visual ML
                </h1>
              </div>
              {user?.isPremium && (
                <span className="px-3 py-1 bg-gradient-to-r from-violet-100 to-violet-50 text-violet-700 text-xs font-semibold rounded-full border border-violet-200 shadow-sm">
                  ✨ Premium
                </span>
              )}
            </div>

            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-3">
                {user?.profilePic ? (
                  <img
                    src={user.profilePic}
                    alt="Profile"
                    className="h-10 w-10 rounded-full ring-2 ring-slate-200 shadow-md"
                  />
                ) : (
                  <div className="h-10 w-10 rounded-full bg-gradient-to-br from-slate-900 to-slate-700 flex items-center justify-center shadow-lg shadow-slate-900/25">
                    <span className="text-white font-semibold">
                      {user?.emailId?.[0]?.toUpperCase()}
                    </span>
                  </div>
                )}
                <div className="hidden sm:block">
                  <p className="text-sm font-medium text-slate-900">
                    {user?.emailId}
                  </p>
                  {user?.collegeOrSchool && (
                    <p className="text-xs text-slate-500">
                      {user.collegeOrSchool}
                    </p>
                  )}
                </div>
              </div>

              <button
                onClick={() => navigate("/profile")}
                className="p-2 text-slate-600 hover:text-slate-900 hover:bg-slate-100/80 rounded-lg transition-all"
                title="Profile Settings"
              >
                <Settings className="h-5 w-5" />
              </button>

              <button
                onClick={handleLogout}
                className="flex items-center space-x-2 px-4 py-2 text-sm font-medium text-red-600 hover:text-red-700 hover:bg-red-50 rounded-lg transition-all border border-red-200/50 hover:border-red-300"
              >
                <LogOut className="h-4 w-4" />
                <span className="hidden sm:inline">Logout</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Quick Stats */}
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
                  Projects
                </p>
                <p className="text-4xl font-bold bg-gradient-to-br from-slate-900 to-slate-700 bg-clip-text text-transparent mt-2">
                  {projects.length}
                </p>
                <div className="flex items-center gap-1.5 mt-2">
                  <TrendingUp className="w-3.5 h-3.5 text-emerald-600" />
                  <span className="text-xs text-emerald-600 font-semibold">
                    Active
                  </span>
                </div>
              </div>
              <div className="relative w-14 h-14 rounded-xl bg-gradient-to-br from-slate-900 to-slate-700 flex items-center justify-center shadow-lg shadow-slate-900/25 group-hover:scale-110 transition-transform">
                <FolderKanban className="h-7 w-7 text-white" />
                <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-white/20 to-transparent" />
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="group bg-white/80 backdrop-blur-xl p-6 rounded-2xl border border-slate-200/60 shadow-lg shadow-slate-900/5 hover:shadow-xl hover:shadow-emerald-500/10 transition-all duration-300 ring-1 ring-slate-900/5"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-500 uppercase tracking-wide">
                  Datasets
                </p>
                <p className="text-4xl font-bold bg-gradient-to-br from-slate-900 to-slate-700 bg-clip-text text-transparent mt-2">
                  {allDatasetsData?.total || 0}
                </p>
                <div className="flex items-center gap-1.5 mt-2">
                  <span className="text-xs text-slate-500">Ready to use</span>
                </div>
              </div>
              <div className="relative w-14 h-14 rounded-xl bg-gradient-to-br from-emerald-500 to-emerald-600 flex items-center justify-center shadow-lg shadow-emerald-500/25 group-hover:scale-110 transition-transform">
                <Database className="h-7 w-7 text-white" />
                <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-white/20 to-transparent" />
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="group bg-white/80 backdrop-blur-xl p-6 rounded-2xl border border-slate-200/60 shadow-lg shadow-slate-900/5 hover:shadow-xl hover:shadow-violet-500/10 transition-all duration-300 ring-1 ring-slate-900/5"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-500 uppercase tracking-wide">
                  Deployed
                </p>
                <p className="text-4xl font-bold bg-gradient-to-br from-slate-900 to-slate-700 bg-clip-text text-transparent mt-2">
                  {recentRuns.length}
                </p>
                <div className="flex items-center gap-1.5 mt-2">
                  <Zap className="w-3.5 h-3.5 text-violet-600" />
                  <span className="text-xs text-violet-600 font-semibold">
                    Live
                  </span>
                </div>
              </div>
              <div className="relative w-14 h-14 rounded-xl bg-gradient-to-br from-violet-500 to-violet-600 flex items-center justify-center shadow-lg shadow-violet-500/25 group-hover:scale-110 transition-transform">
                <PlayCircle className="h-7 w-7 text-white" />
                <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-white/20 to-transparent" />
              </div>
            </div>
          </motion.div>
        </div>

        {/* Projects Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="bg-white/80 backdrop-blur-xl rounded-2xl border border-slate-200/60 shadow-xl shadow-slate-900/5 ring-1 ring-slate-900/5"
        >
          <div className="p-6 border-b border-slate-200/60">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold bg-gradient-to-r from-slate-900 to-slate-700 bg-clip-text text-transparent">
                My Projects
              </h2>
              <button
                onClick={() => setIsCreateModalOpen(true)}
                className="group flex items-center space-x-2 px-5 py-2.5 bg-slate-900 text-white rounded-xl hover:bg-slate-800 transition-all shadow-lg shadow-slate-900/25 hover:shadow-xl hover:shadow-slate-900/40 hover:-translate-y-0.5"
              >
                <Plus className="h-4 w-4 group-hover:rotate-90 transition-transform" />
                <span className="font-medium">New Project</span>
              </button>
            </div>

            {/* Search Bar */}
            <div className="relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-slate-400" />
              <input
                type="text"
                placeholder="Search projects..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-12 pr-4 py-3 bg-white/60 border border-slate-200/60 rounded-xl focus:outline-none focus:ring-2 focus:ring-slate-900 focus:border-transparent transition-all shadow-sm hover:shadow-md backdrop-blur-sm"
              />
            </div>
          </div>

          <div className="p-6">
            {filteredProjects.length === 0 ? (
              <div className="text-center py-16">
                <div className="relative inline-block">
                  <div className="absolute inset-0 bg-gradient-to-br from-slate-200 to-slate-100 rounded-full blur-2xl opacity-50"></div>
                  <div className="relative w-20 h-20 mx-auto bg-gradient-to-br from-slate-100 to-slate-50 rounded-2xl flex items-center justify-center shadow-lg">
                    <FolderKanban className="h-10 w-10 text-slate-400" />
                  </div>
                </div>
                <h3 className="mt-6 text-xl font-bold text-slate-900">
                  No projects yet
                </h3>
                <p className="mt-2 text-sm text-slate-600 max-w-sm mx-auto">
                  Get started by creating your first ML project and bring your
                  ideas to life
                </p>
                <button
                  onClick={() => setIsCreateModalOpen(true)}
                  className="mt-8 inline-flex items-center space-x-2 px-6 py-3 bg-slate-900 text-white rounded-xl hover:bg-slate-800 transition-all shadow-lg shadow-slate-900/25 hover:shadow-xl hover:shadow-slate-900/40 hover:-translate-y-0.5"
                >
                  <Plus className="h-5 w-5" />
                  <span className="font-medium">Create Project</span>
                </button>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredProjects.map((project, index) => (
                  <motion.div
                    key={project.id}
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                    onClick={() => handleProjectClick(project.id)}
                    className="group p-6 bg-white/60 backdrop-blur-sm border border-slate-200/60 rounded-2xl hover:border-slate-300 hover:shadow-xl hover:shadow-slate-900/10 transition-all cursor-pointer relative ring-1 ring-slate-900/5 hover:-translate-y-1"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-slate-900 to-slate-700 flex items-center justify-center shadow-md shadow-slate-900/25">
                            <FolderKanban className="w-4 h-4 text-white" />
                          </div>
                          <h3 className="font-bold text-slate-900 group-hover:text-slate-900">
                            {project.name}
                          </h3>
                        </div>
                      </div>
                      <button
                        onClick={(e) => handleDeleteProject(e, project.id)}
                        className="opacity-0 group-hover:opacity-100 p-2 text-red-600 hover:bg-red-50 rounded-lg transition-all hover:scale-110 border border-red-200"
                        title="Delete project"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                    <p className="text-sm text-slate-600 mb-4 line-clamp-2 leading-relaxed">
                      {project.description || "No description"}
                    </p>
                    <div className="flex items-center justify-between pt-3 border-t border-slate-200/60">
                      <span className="text-xs text-slate-500 font-medium">
                        Updated{" "}
                        {new Date(project.updatedAt).toLocaleDateString()}
                      </span>
                      <div className="flex items-center gap-1 text-slate-400 group-hover:text-slate-900 transition-colors">
                        <span className="text-xs">Open</span>
                        <svg
                          className="w-3 h-3 group-hover:translate-x-0.5 transition-transform"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M9 5l7 7-7 7"
                          />
                        </svg>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </div>
        </motion.div>

        {/* Create Project Modal */}
        <CreateProjectModal
          isOpen={isCreateModalOpen}
          onClose={() => setIsCreateModalOpen(false)}
          onSuccess={handleCreateProject}
        />

        {/* Recent Runs */}
        {recentRuns.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.5 }}
            className="mt-8 bg-white/80 backdrop-blur-xl rounded-2xl border border-slate-200/60 shadow-xl shadow-slate-900/5 ring-1 ring-slate-900/5"
          >
            <div className="p-6 border-b border-slate-200/60">
              <h2 className="text-2xl font-bold bg-gradient-to-r from-slate-900 to-slate-700 bg-clip-text text-transparent">
                Recent Runs
              </h2>
            </div>
            <div className="p-6">
              <div className="space-y-3">
                {recentRuns.map((run, index) => (
                  <motion.div
                    key={run.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                    className="flex items-center justify-between p-4 bg-white/60 backdrop-blur-sm border border-slate-200/60 rounded-xl hover:shadow-lg hover:shadow-slate-900/5 transition-all ring-1 ring-slate-900/5"
                  >
                    <div className="flex items-center space-x-3">
                      {getStatusIcon(run.status)}
                      <div>
                        <p className="text-sm font-semibold text-slate-900">
                          Run #{run.id.slice(0, 8)}
                        </p>
                        <p className="text-xs text-slate-500">
                          {new Date(run.createdAt).toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <span
                      className={`px-3 py-1.5 text-xs font-semibold rounded-full border ${
                        run.status === "completed"
                          ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                          : run.status === "failed"
                            ? "bg-red-50 text-red-700 border-red-200"
                            : run.status === "running"
                              ? "bg-blue-50 text-blue-700 border-blue-200"
                              : "bg-slate-50 text-slate-700 border-slate-200"
                      }`}
                    >
                      {run.status}
                    </span>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </main>
    </div>
  );
};

export default StudentDashboard;
