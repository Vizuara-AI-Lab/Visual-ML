import React, { useState } from "react";
import { useNavigate } from "react-router";
import {
  LogOut,
  FolderKanban,
  Database,
  Share2,
  Settings,
  Plus,
  Search,
  Trash2,
  ChevronRight,
  Clock,
} from "lucide-react";
import { motion } from "framer-motion";
import { useProjects } from "../../hooks/queries/useProjects";
import { useDeleteProject } from "../../hooks/mutations/useDeleteProject";
import { CreateProjectModal } from "../../components/projects/CreateProjectModal";
import { useAllDatasets } from "../../hooks/queries/useAllDatasets";

interface User {
  emailId: string;
  fullName?: string;
  collegeOrSchool?: string;
  profilePic?: string;
  isPremium: boolean;
}

const StudentDashboard: React.FC = () => {
  const navigate = useNavigate();
  const [user, setUser] = useState<User | null>(null);
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

  const sharedProjectsCount = projects.filter((p) => p.is_public).length;

  const filteredProjects = projects.filter((project) =>
    project.name.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <div className="animate-spin rounded-full h-12 w-12 border-[3px] border-slate-200 border-t-slate-800"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="sticky top-0 z-40 bg-white border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Left: Logo + Title */}
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-slate-900 flex items-center justify-center">
                <FolderKanban className="w-4 h-4 text-white" />
              </div>
              <h1 className="text-xl font-bold text-slate-900">Visual ML</h1>
              {user?.isPremium && (
                <span className="px-2.5 py-0.5 bg-amber-100 text-amber-700 text-xs font-semibold rounded-full border border-amber-200">
                  Premium
                </span>
              )}
            </div>

            {/* Center: Inline Stats (visible on md+) */}
            <div className="hidden md:flex items-center gap-6">
              <div className="flex items-center gap-2">
                <FolderKanban className="w-4 h-4 text-slate-400" />
                <span className="text-sm text-slate-500">Projects</span>
                <span className="text-sm font-semibold text-slate-900">
                  {projects.length}
                </span>
              </div>
              <div className="w-px h-4 bg-slate-200" />
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-slate-400" />
                <span className="text-sm text-slate-500">Datasets</span>
                <span className="text-sm font-semibold text-slate-900">
                  {allDatasetsData?.total || 0}
                </span>
              </div>
              <div className="w-px h-4 bg-slate-200" />
              <div className="flex items-center gap-2">
                <Share2 className="w-4 h-4 text-slate-400" />
                <span className="text-sm text-slate-500">Shared</span>
                <span className="text-sm font-semibold text-slate-900">
                  {sharedProjectsCount}
                </span>
              </div>
            </div>

            {/* Right: User info + actions */}
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-3">
                {user?.profilePic ? (
                  <img
                    src={user.profilePic}
                    alt="Profile"
                    className="h-8 w-8 rounded-full ring-2 ring-slate-100"
                  />
                ) : (
                  <div className="h-8 w-8 rounded-full bg-slate-800 flex items-center justify-center">
                    <span className="text-white text-sm font-semibold">
                      {user?.emailId?.[0]?.toUpperCase()}
                    </span>
                  </div>
                )}
                <div className="hidden sm:block">
                  <p className="text-sm font-medium text-slate-900 leading-tight">
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
                className="p-2 text-slate-500 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition-colors"
                title="Profile Settings"
              >
                <Settings className="h-4 w-4" />
              </button>

              <button
                onClick={handleLogout}
                className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-red-600 hover:text-red-700 hover:bg-red-50 rounded-lg transition-colors border border-red-200"
              >
                <LogOut className="h-4 w-4" />
                <span className="hidden sm:inline">Logout</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Mobile stats strip */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="md:hidden flex items-center justify-around py-3 mb-6 bg-white rounded-lg border border-slate-200"
        >
          <div className="text-center">
            <p className="text-lg font-bold text-slate-900">
              {projects.length}
            </p>
            <p className="text-xs text-slate-500">Projects</p>
          </div>
          <div className="w-px h-8 bg-slate-200" />
          <div className="text-center">
            <p className="text-lg font-bold text-slate-900">
              {allDatasetsData?.total || 0}
            </p>
            <p className="text-xs text-slate-500">Datasets</p>
          </div>
          <div className="w-px h-8 bg-slate-200" />
          <div className="text-center">
            <p className="text-lg font-bold text-slate-900">
              {sharedProjectsCount}
            </p>
            <p className="text-xs text-slate-500">Shared</p>
          </div>
        </motion.div>

        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
          className="mb-8 bg-white border border-amber-100 rounded-lg px-6 py-8 sm:px-8 sm:py-10 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-6"
        >
          <div>
            <p className="text-sm font-medium text-amber-600/70 mb-1">Dashboard</p>
            <h2 className="text-2xl sm:text-3xl font-bold text-slate-900">
              Welcome back, {user?.fullName?.split(" ")[0] || user?.emailId?.split("@")[0] || "there"}
            </h2>
            <p className="mt-2 text-sm sm:text-base text-slate-500 max-w-lg">
              Build, train, and visualize machine learning models with an intuitive
              drag-and-drop pipeline. Pick up where you left off or start something new.
            </p>
          </div>
          <button
            onClick={() => setIsCreateModalOpen(true)}
            className="flex items-center gap-2 px-5 py-2.5 bg-slate-900 text-white text-sm font-medium rounded-lg hover:bg-slate-800 transition-colors shrink-0 self-start sm:self-center"
          >
            <Plus className="h-4 w-4" />
            <span>New Project</span>
          </button>
        </motion.div>

        {/* Projects section header */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-slate-900">
              My Projects
            </h2>
            <span className="text-sm text-slate-400">
              {filteredProjects.length} {filteredProjects.length === 1 ? "project" : "projects"}
            </span>
          </div>

          {/* Search Bar */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
            <input
              type="text"
              placeholder="Search projects..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 bg-white border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-amber-400 focus:border-amber-400 transition-colors"
            />
          </div>
        </div>

        {/* Project list */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
        >
          {filteredProjects.length === 0 ? (
            <div className="text-center py-16">
              <div className="w-16 h-16 mx-auto bg-amber-50 rounded-xl flex items-center justify-center mb-4">
                <FolderKanban className="h-8 w-8 text-amber-400" />
              </div>
              <h3 className="text-lg font-semibold text-slate-900">
                No projects yet
              </h3>
              <p className="mt-1 text-sm text-slate-500 max-w-sm mx-auto">
                Get started by creating your first ML project and bring your
                ideas to life
              </p>
              <button
                onClick={() => setIsCreateModalOpen(true)}
                className="mt-6 inline-flex items-center gap-2 px-5 py-2.5 bg-slate-900 text-white text-sm font-medium rounded-lg hover:bg-slate-800 transition-colors"
              >
                <Plus className="h-4 w-4" />
                <span>Create Project</span>
              </button>
            </div>
          ) : (
            <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
              {filteredProjects.map((project, index) => (
                <motion.div
                  key={project.id}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{
                    duration: 0.2,
                    delay: Math.min(index * 0.03, 0.5),
                  }}
                  onClick={() => handleProjectClick(project.id)}
                  className={`group flex items-center gap-4 px-4 py-3.5 hover:bg-slate-50 transition-colors cursor-pointer ${
                    index !== 0 ? "border-t border-slate-100" : ""
                  }`}
                >
                  {/* Project icon */}
                  <div className="w-9 h-9 rounded-lg bg-amber-50 flex items-center justify-center shrink-0">
                    <FolderKanban className="w-4 h-4 text-amber-600" />
                  </div>

                  {/* Project info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <h3 className="text-sm font-semibold text-slate-900 truncate">
                        {project.name}
                      </h3>
                      {project.is_public && (
                        <span className="px-1.5 py-0.5 text-[10px] font-medium text-emerald-700 bg-emerald-50 border border-emerald-200 rounded shrink-0">
                          Public
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-slate-500 truncate mt-0.5">
                      {project.description || "No description"}
                    </p>
                  </div>

                  {/* Updated date */}
                  <div className="hidden sm:flex items-center gap-1.5 text-xs text-slate-400 shrink-0">
                    <Clock className="w-3.5 h-3.5" />
                    <span>
                      {new Date(project.updatedAt).toLocaleDateString()}
                    </span>
                  </div>

                  {/* Delete button */}
                  <button
                    onClick={(e) => handleDeleteProject(e, project.id)}
                    className="opacity-0 group-hover:opacity-100 p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-md transition-all shrink-0"
                    title="Delete project"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>

                  {/* Arrow indicator */}
                  <ChevronRight className="w-4 h-4 text-slate-300 group-hover:text-slate-500 transition-colors shrink-0" />
                </motion.div>
              ))}
            </div>
          )}
        </motion.div>

        {/* Create Project Modal */}
        <CreateProjectModal
          isOpen={isCreateModalOpen}
          onClose={() => setIsCreateModalOpen(false)}
          onSuccess={handleCreateProject}
        />
      </main>
    </div>
  );
};

export default StudentDashboard;
