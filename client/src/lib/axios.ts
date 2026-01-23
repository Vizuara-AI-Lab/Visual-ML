import axios from "axios";
import { env } from "./env";

const axiosInstance = axios.create({
  baseURL: env.API_URL,
  withCredentials: true,  // ✅ CRITICAL: Send cookies with every request
  headers: {
    "Content-Type": "application/json",
  },
});

// ❌ REMOVED: Authorization header interceptor (cookies are sent automatically)
// No need to manually add tokens to headers

// Response interceptor to handle token refresh
axiosInstance.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    // Don't attempt token refresh for auth endpoints (login, register, etc.)
    const isAuthEndpoint =
      originalRequest.url?.includes("/auth/student/login") ||
      originalRequest.url?.includes("/auth/student/register") ||
      originalRequest.url?.includes("/auth/student/google") ||
      originalRequest.url?.includes("/auth/admin/login") ||
      originalRequest.url?.includes("/auth/refresh");

    // If error is 401 and we haven't retried yet, and it's not an auth endpoint
    if (
      error.response?.status === 401 &&
      !originalRequest._retry &&
      !isAuthEndpoint
    ) {
      originalRequest._retry = true;

      try {
        // Try to refresh token (cookies are sent automatically)
        await axiosInstance.post("/auth/refresh");

        // ✅ New cookies are set automatically by backend
        // Retry original request (cookies will be sent automatically)
        return axiosInstance(originalRequest);
      } catch (refreshError) {
        // Refresh failed, logout user
        localStorage.removeItem("user");
        window.location.href = "/auth/signin";
        return Promise.reject(refreshError);
      }
    }

    return Promise.reject(error);
  },
);

export default axiosInstance;
