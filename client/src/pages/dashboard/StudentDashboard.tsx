import React, { useState, useEffect } from "react";
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
} from "lucide-react";
import axiosInstance from "../../lib/axios";

interface User {
  emailId: string;
  collegeOrSchool?: string;
  profilePic?: string;
  isPremium: boolean;
}

interface Project {
  id: string;
  name: string;
  description?: string;
  createdAt: string;
  updatedAt: string;
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
  const [projects, setProjects] = useState<Project[]>([]);
  const [recentRuns, setRecentRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    loadUserData();
    loadProjects();
    loadRecentRuns();
  }, []);

  const loadUserData = () => {
    const userData = localStorage.getItem("user");
    if (userData) {
      setUser(JSON.parse(userData));
    }
  };

  const loadProjects = async () => {
    try {
      const response = await axiosInstance.get("/projects");
      setProjects(response.data || []);
    } catch (error) {
      console.error("Failed to load projects:", error);
      setProjects([]);
    } finally {
      setLoading(false);
    }
  };

  const loadRecentRuns = async () => {
    try {
      const response = await axiosInstance.get("/runs/recent");
      setRecentRuns(response.data || []);
    } catch (error) {
      console.error("Failed to load recent runs:", error);
      setRecentRuns([]);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("accessToken");
    localStorage.removeItem("refreshToken");
    localStorage.removeItem("user");
    navigate("/signin");
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

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <h1 className="text-2xl font-bold text-gray-900">Visual ML</h1>
              {user?.isPremium && (
                <span className="px-3 py-1 bg-blue-100 text-blue-800 text-xs font-semibold rounded-full">
                  Premium
                </span>
              )}
            </div>

            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-3">
                {user?.profilePic ? (
                  <img
                    src={user.profilePic}
                    alt="Profile"
                    className="h-10 w-10 rounded-full"
                  />
                ) : (
                  <div className="h-10 w-10 rounded-full bg-blue-600 flex items-center justify-center">
                    <span className="text-white font-semibold">
                      {user?.emailId?.[0]?.toUpperCase()}
                    </span>
                  </div>
                )}
                <div className="hidden sm:block">
                  <p className="text-sm font-medium text-gray-900">
                    {user?.emailId}
                  </p>
                  {user?.collegeOrSchool && (
                    <p className="text-xs text-gray-500">
                      {user.collegeOrSchool}
                    </p>
                  )}
                </div>
              </div>

              <button
                onClick={() => navigate("/settings")}
                className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <Settings className="h-5 w-5" />
              </button>

              <button
                onClick={handleLogout}
                className="flex items-center space-x-2 px-4 py-2 text-sm font-medium text-red-600 hover:bg-red-50 rounded-lg transition-colors"
              >
                <LogOut className="h-4 w-4" />
                <span>Logout</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Projects</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">
                  {projects.length}
                </p>
              </div>
              <div className="p-3 bg-blue-100 rounded-lg">
                <FolderKanban className="h-6 w-6 text-blue-600" />
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Datasets</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">0</p>
              </div>
              <div className="p-3 bg-green-100 rounded-lg">
                <Database className="h-6 w-6 text-green-600" />
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Runs</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">
                  {recentRuns.length}
                </p>
              </div>
              <div className="p-3 bg-purple-100 rounded-lg">
                <PlayCircle className="h-6 w-6 text-purple-600" />
              </div>
            </div>
          </div>
        </div>

        {/* Projects Section */}
        <div className="bg-white rounded-lg border border-gray-200">
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-900">My Projects</h2>
              <button
                onClick={() => navigate("/projects/new")}
                className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                <Plus className="h-4 w-4" />
                <span>New Project</span>
              </button>
            </div>

            {/* Search Bar */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search projects..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>

          <div className="p-6">
            {filteredProjects.length === 0 ? (
              <div className="text-center py-12">
                <FolderKanban className="mx-auto h-12 w-12 text-gray-400" />
                <h3 className="mt-4 text-lg font-medium text-gray-900">
                  No projects yet
                </h3>
                <p className="mt-2 text-sm text-gray-600">
                  Get started by creating your first ML project
                </p>
                <button
                  onClick={() => navigate("/projects/new")}
                  className="mt-6 inline-flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  <Plus className="h-4 w-4" />
                  <span>Create Project</span>
                </button>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {filteredProjects.map((project) => (
                  <div
                    key={project.id}
                    onClick={() => navigate(`/projects/${project.id}`)}
                    className="p-4 border border-gray-200 rounded-lg hover:border-blue-500 hover:shadow-md transition-all cursor-pointer"
                  >
                    <h3 className="font-semibold text-gray-900 mb-2">
                      {project.name}
                    </h3>
                    <p className="text-sm text-gray-600 mb-4 line-clamp-2">
                      {project.description || "No description"}
                    </p>
                    <div className="flex items-center justify-between text-xs text-gray-500">
                      <span>
                        Updated{" "}
                        {new Date(project.updatedAt).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Recent Runs */}
        {recentRuns.length > 0 && (
          <div className="mt-8 bg-white rounded-lg border border-gray-200">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-xl font-bold text-gray-900">Recent Runs</h2>
            </div>
            <div className="p-6">
              <div className="space-y-3">
                {recentRuns.map((run) => (
                  <div
                    key={run.id}
                    className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    <div className="flex items-center space-x-3">
                      {getStatusIcon(run.status)}
                      <div>
                        <p className="text-sm font-medium text-gray-900">
                          Run #{run.id.slice(0, 8)}
                        </p>
                        <p className="text-xs text-gray-500">
                          {new Date(run.createdAt).toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <span
                      className={`px-3 py-1 text-xs font-semibold rounded-full ${
                        run.status === "completed"
                          ? "bg-green-100 text-green-800"
                          : run.status === "failed"
                            ? "bg-red-100 text-red-800"
                            : run.status === "running"
                              ? "bg-blue-100 text-blue-800"
                              : "bg-gray-100 text-gray-800"
                      }`}
                    >
                      {run.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default StudentDashboard;
