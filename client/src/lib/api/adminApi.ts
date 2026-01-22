import axiosInstance from '../axios';
import type { AdminProfile, StudentListItem, Student, StudentFilters, UpdateStudentData } from '../../types/api';

// Get admin profile
export const getAdminProfile = async (): Promise<AdminProfile> => {
  const { data } = await axiosInstance.get('/auth/admin/me');
  return data;
};

// Get students list with filters
export const getStudentsList = async (filters: StudentFilters = {}): Promise<StudentListItem[]> => {
  const params = new URLSearchParams();
  
  if (filters.skip !== undefined) params.append('skip', filters.skip.toString());
  if (filters.limit !== undefined) params.append('limit', filters.limit.toString());
  if (filters.search) params.append('search', filters.search);
  if (filters.isPremium !== undefined) params.append('isPremium', filters.isPremium.toString());
  if (filters.isActive !== undefined) params.append('isActive', filters.isActive.toString());

  const { data } = await axiosInstance.get(`/auth/admin/students?${params}`);
  return data;
};

// Get student by ID
export const getStudentById = async (id: string | number): Promise<Student> => {
  const { data } = await axiosInstance.get(`/auth/admin/students/${id}`);
  return data;
};

// Update student
export const updateStudent = async (id: string | number, updates: UpdateStudentData): Promise<Student> => {
  const { data } = await axiosInstance.patch(`/auth/admin/students/${id}`, updates);
  return data;
};
