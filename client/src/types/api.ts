// API Response Types

export interface Student {
  id: number;
  emailId: string;
  fullName: string;
  role: string;
  authProvider: "LOCAL" | "GOOGLE";
  collegeOrSchool?: string;
  contactNo?: string;
  recentProject?: string;
  profilePic?: string;
  isPremium: boolean;
  isActive: boolean;
  xp: number;
  level: number;
  createdAt: string;
  lastLogin?: string;
}

export interface Badge {
  badge_id: string;
  name: string;
  description: string;
  icon: string;
  color: string;
  awarded_at?: string;
  is_earned: boolean;
}

export interface GamificationProfile {
  xp: number;
  level: number;
  xp_to_next_level: number;
  progress_percent: number;
  total_badges_earned: number;
  badges: Badge[];
}

export interface AwardXPResponse {
  xp_gained: number;
  total_xp: number;
  level: number;
  leveled_up: boolean;
  new_level?: number;
  new_badges: Badge[];
}

export interface StudentListItem {
  id: number;
  emailId: string;
  fullName: string;
  authProvider: "LOCAL" | "GOOGLE";
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
  role: "ADMIN";
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
