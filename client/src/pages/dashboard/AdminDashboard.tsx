import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router";
import axiosInstance from "../../lib/axios";

interface Student {
  id: number;
  emailId: string;
  collegeOrSchool?: string;
  isPremium: boolean;
  isActive: boolean;
  createdAt: string;
}

const AdminDashboard: React.FC = () => {
  const navigate = useNavigate();
  const [students, setStudents] = useState<Student[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState({ isPremium: "", isActive: "" });

  useEffect(() => {
    loadStudents();
  }, []);

  const loadStudents = async () => {
    try {
      const params = new URLSearchParams();
      if (search) params.append("search", search);
      if (filter.isPremium) params.append("isPremium", filter.isPremium);
      if (filter.isActive) params.append("isActive", filter.isActive);

      const response = await axiosInstance.get(
        `/auth/admin/students?${params}`,
      );
      setStudents(response.data);
    } catch (err) {
      console.error("Failed to load students:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = () => {
    loadStudents();
  };

  const handleLogout = () => {
    localStorage.clear();
    navigate("/auth/signin");
  };

  const togglePremium = async (studentId: number, currentStatus: boolean) => {
    try {
      await axiosInstance.patch(`/auth/admin/students/${studentId}`, {
        isPremium: !currentStatus,
      });
      loadStudents();
    } catch (err) {
      console.error("Failed to update premium status:", err);
    }
  };

  const toggleActive = async (studentId: number, currentStatus: boolean) => {
    try {
      await axiosInstance.patch(`/auth/admin/students/${studentId}`, {
        isActive: !currentStatus,
      });
      loadStudents();
    } catch (err) {
      console.error("Failed to update active status:", err);
    }
  };

  if (loading) {
    return <div className="p-4">Loading...</div>;
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white border-b p-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <h1 className="text-2xl font-bold">Admin Dashboard</h1>
          <button
            onClick={handleLogout}
            className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
          >
            Logout
          </button>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-4">
        <div className="bg-white p-4 rounded shadow mb-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <input
              type="text"
              placeholder="Search by email or college"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="px-3 py-2 border rounded"
            />

            <select
              value={filter.isPremium}
              onChange={(e) =>
                setFilter({ ...filter, isPremium: e.target.value })
              }
              className="px-3 py-2 border rounded"
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
              className="px-3 py-2 border rounded"
            >
              <option value="">All Status</option>
              <option value="true">Active Only</option>
              <option value="false">Inactive Only</option>
            </select>

            <button
              onClick={handleSearch}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Search
            </button>
          </div>
        </div>

        <div className="bg-white rounded shadow">
          <div className="p-4 border-b">
            <h2 className="text-xl font-bold">Students ({students.length})</h2>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-100">
                <tr>
                  <th className="p-3 text-left">ID</th>
                  <th className="p-3 text-left">Email</th>
                  <th className="p-3 text-left">College/School</th>
                  <th className="p-3 text-left">Premium</th>
                  <th className="p-3 text-left">Status</th>
                  <th className="p-3 text-left">Joined</th>
                  <th className="p-3 text-left">Actions</th>
                </tr>
              </thead>
              <tbody>
                {students.map((student) => (
                  <tr key={student.id} className="border-b hover:bg-gray-50">
                    <td className="p-3">{student.id}</td>
                    <td className="p-3">{student.emailId}</td>
                    <td className="p-3">{student.collegeOrSchool || "-"}</td>
                    <td className="p-3">
                      <span
                        className={`px-2 py-1 rounded text-xs ${
                          student.isPremium
                            ? "bg-yellow-100 text-yellow-800"
                            : "bg-gray-100 text-gray-800"
                        }`}
                      >
                        {student.isPremium ? "Premium" : "Free"}
                      </span>
                    </td>
                    <td className="p-3">
                      <span
                        className={`px-2 py-1 rounded text-xs ${
                          student.isActive
                            ? "bg-green-100 text-green-800"
                            : "bg-red-100 text-red-800"
                        }`}
                      >
                        {student.isActive ? "Active" : "Inactive"}
                      </span>
                    </td>
                    <td className="p-3">
                      {new Date(student.createdAt).toLocaleDateString()}
                    </td>
                    <td className="p-3">
                      <div className="flex gap-2">
                        <button
                          onClick={() =>
                            navigate(`/admin/students/${student.id}`)
                          }
                          className="px-2 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-700"
                        >
                          View
                        </button>
                        <button
                          onClick={() =>
                            togglePremium(student.id, student.isPremium)
                          }
                          className="px-2 py-1 bg-yellow-600 text-white text-xs rounded hover:bg-yellow-700"
                        >
                          {student.isPremium ? "Revoke" : "Grant"} Premium
                        </button>
                        <button
                          onClick={() =>
                            toggleActive(student.id, student.isActive)
                          }
                          className={`px-2 py-1 text-white text-xs rounded ${
                            student.isActive
                              ? "bg-red-600 hover:bg-red-700"
                              : "bg-green-600 hover:bg-green-700"
                          }`}
                        >
                          {student.isActive ? "Deactivate" : "Activate"}
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {students.length === 0 && (
            <div className="p-8 text-center text-gray-500">
              No students found
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;
