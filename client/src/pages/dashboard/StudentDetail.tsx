import React, { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import axiosInstance from "../../lib/axios";

interface StudentDetail {
  id: number;
  emailId: string;
  role: string;
  authProvider: string;
  collegeOrSchool?: string;
  contactNo?: string;
  recentProject?: string;
  profilePic?: string;
  isPremium: boolean;
  isActive: boolean;
  createdAt: string;
  lastLogin?: string;
}

const StudentDetail: React.FC = () => {
  const navigate = useNavigate();
  const { id } = useParams();
  const [student, setStudent] = useState<StudentDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    loadStudent();
  }, [id]);

  const loadStudent = async () => {
    try {
      const response = await axiosInstance.get(`/auth/admin/students/${id}`);
      setStudent(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to load student");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="p-4">Loading...</div>;
  }

  if (error) {
    return (
      <div className="p-4">
        <div className="bg-red-100 text-red-800 p-4 rounded">{error}</div>
        <button
          onClick={() => navigate("/admin/dashboard")}
          className="mt-4 px-4 py-2 bg-gray-600 text-white rounded"
        >
          Back to Dashboard
        </button>
      </div>
    );
  }

  if (!student) {
    return <div className="p-4">Student not found</div>;
  }

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-3xl mx-auto bg-white p-6 rounded shadow">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold">Student Details</h1>
          <button
            onClick={() => navigate("/admin/dashboard")}
            className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
          >
            Back to Dashboard
          </button>
        </div>

        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="font-bold text-gray-600">ID</p>
              <p>{student.id}</p>
            </div>
            <div>
              <p className="font-bold text-gray-600">Email</p>
              <p>{student.emailId}</p>
            </div>
            <div>
              <p className="font-bold text-gray-600">Role</p>
              <p>{student.role}</p>
            </div>
            <div>
              <p className="font-bold text-gray-600">Auth Provider</p>
              <p>{student.authProvider}</p>
            </div>
            <div>
              <p className="font-bold text-gray-600">College/School</p>
              <p>{student.collegeOrSchool || "-"}</p>
            </div>
            <div>
              <p className="font-bold text-gray-600">Contact Number</p>
              <p>{student.contactNo || "-"}</p>
            </div>
            <div>
              <p className="font-bold text-gray-600">Premium Status</p>
              <p>
                <span
                  className={`px-2 py-1 rounded text-xs ${
                    student.isPremium
                      ? "bg-yellow-100 text-yellow-800"
                      : "bg-gray-100 text-gray-800"
                  }`}
                >
                  {student.isPremium ? "Premium" : "Free"}
                </span>
              </p>
            </div>
            <div>
              <p className="font-bold text-gray-600">Account Status</p>
              <p>
                <span
                  className={`px-2 py-1 rounded text-xs ${
                    student.isActive
                      ? "bg-green-100 text-green-800"
                      : "bg-red-100 text-red-800"
                  }`}
                >
                  {student.isActive ? "Active" : "Inactive"}
                </span>
              </p>
            </div>
            <div>
              <p className="font-bold text-gray-600">Joined</p>
              <p>{new Date(student.createdAt).toLocaleString()}</p>
            </div>
            <div>
              <p className="font-bold text-gray-600">Last Login</p>
              <p>
                {student.lastLogin
                  ? new Date(student.lastLogin).toLocaleString()
                  : "Never"}
              </p>
            </div>
          </div>

          {student.recentProject && (
            <div>
              <p className="font-bold text-gray-600">Recent Project</p>
              <p className="mt-2 p-3 bg-gray-100 rounded">
                {student.recentProject}
              </p>
            </div>
          )}

          {student.profilePic && (
            <div>
              <p className="font-bold text-gray-600">Profile Picture</p>
              <img
                src={student.profilePic}
                alt="Profile"
                className="mt-2 w-32 h-32 rounded object-cover"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default StudentDetail;
