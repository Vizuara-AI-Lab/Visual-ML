import axiosInstance from '../axios';

export interface Project {
  id: number;
  name: string;
  description?: string;
  studentId: number;
  createdAt: string;
  updatedAt: string;
  lastRunAt?: string;
}

export interface ProjectListItem {
  id: number;
  name: string;
  description?: string;
  createdAt: string;
  updatedAt: string;
  is_public?: boolean;
  share_token?: string;
}

export interface ProjectState {
  nodes: any[];
  edges: any[];
  datasetMetadata?: any;
  executionResult?: any;
}

export interface CreateProjectData {
  name: string;
  description?: string;
}

export interface UpdateProjectData {
  name?: string;
  description?: string;
}

// Create new project
export const createProject = async (data: CreateProjectData): Promise<Project> => {
  const { data: response } = await axiosInstance.post('/projects', data);
  return response;
};

// Get all projects for current student
export const getProjects = async (): Promise<ProjectListItem[]> => {
  const { data } = await axiosInstance.get('/projects');
  return data;
};

// Get project by ID
export const getProjectById = async (id: string | number): Promise<Project> => {
  const { data } = await axiosInstance.get(`/projects/${id}`);
  return data;
};

// Update project
export const updateProject = async (
  id: string | number,
  updates: UpdateProjectData
): Promise<Project> => {
  const { data } = await axiosInstance.patch(`/projects/${id}`, updates);
  return data;
};

// Delete project
export const deleteProject = async (id: string | number): Promise<void> => {
  await axiosInstance.delete(`/projects/${id}`);
};

// Save project state
export const saveProjectState = async (
  id: string | number,
  state: ProjectState
): Promise<{ projectId: number; state: ProjectState; updatedAt: string }> => {
  const { data } = await axiosInstance.post(`/projects/${id}/save`, state);
  return data;
};

// Get project state
export const getProjectState = async (
  id: string | number
): Promise<{ projectId: number; state: ProjectState; updatedAt: string }> => {
  const { data } = await axiosInstance.get(`/projects/${id}/state`);
  return data;
};
