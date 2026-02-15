/**
 * Zustand store for authentication state
 */

import { create } from "zustand";
import type { Student } from "../types/api";

interface AuthStore {
  user: Student | null;
  setUser: (user: Student | null) => void;
  clearUser: () => void;
  loadUserFromStorage: () => void;
}

export const useAuthStore = create<AuthStore>((set) => ({
  user: null,

  setUser: (user) => {
    set({ user });
    if (user) {
      localStorage.setItem("user", JSON.stringify(user));
    } else {
      localStorage.removeItem("user");
    }
  },

  clearUser: () => {
    set({ user: null });
    localStorage.removeItem("user");
    localStorage.removeItem("userRole");
  },

  loadUserFromStorage: () => {
    try {
      const userData = localStorage.getItem("user");
      if (userData) {
        const user = JSON.parse(userData) as Student;
        set({ user });
      }
    } catch (error) {
      console.error("Failed to load user from storage:", error);
      localStorage.removeItem("user");
    }
  },
}));

// Initialize user from localStorage when the store is created
useAuthStore.getState().loadUserFromStorage();
