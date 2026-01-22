import axiosInstance from '../axios';
import type { Student } from '../../types/api';

// Get current student profile
export const getStudentProfile = async (): Promise<Student> => {
  const { data } = await axiosInstance.get('/auth/student/me');
  return data;
};

// Get projects (placeholder - adjust based on actual API)
export const getProjects = async (): Promise<any[]> => {
  const { data } = await axiosInstance.get('/projects');
  return data;
};

// Get recent runs (placeholder - adjust based on actual API)
export const getRecentRuns = async (): Promise<any[]> => {
  const { data } = await axiosInstance.get('/runs/recent');
  return data;
};
