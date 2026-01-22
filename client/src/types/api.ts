// API Response Types

export interface Student {
  id: number;
  emailId: string;
  role: string;
  authProvider: 'LOCAL' | 'GOOGLE';
  collegeOrSchool?: string;
  contactNo?: string;
  recentProject?: string;
  profilePic?: string;
  isPremium: boolean;
  isActive: boolean;
  createdAt: string;
  lastLogin?: string;
}

export interface StudentListItem {
  id: number;
  emailId: string;
  authProvider: 'LOCAL' | 'GOOGLE';
  collegeOrSchool?: string;
  isPremium: boolean;
  isActive: boolean;
  createdAt: string;
  lastLogin?: string;
}

export interface AdminProfile {
  id: number;
  email: string;
  name?: string;
  role: 'ADMIN';
  isActive: boolean;
  createdAt: string;
  lastLogin?: string;
}

export interface TokenResponse {
  accessToken: string;
  refreshToken: string;
  tokenType: string;
  expiresIn: number;
}

export interface AuthResponse {
  user: Student | AdminProfile;
  tokens: TokenResponse;
  message: string;
}

// Filter types
export interface StudentFilters {
  skip?: number;
  limit?: number;
  search?: string;
  isPremium?: boolean;
  isActive?: boolean;
}

// Mutation types
export interface AdminLoginData {
  email: string;
  password: string;
}

export interface UpdateStudentData {
  isPremium?: boolean;
  isActive?: boolean;
}
