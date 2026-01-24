import axios from "axios";
import { env } from "./env";

const axiosInstance = axios.create({
  baseURL: env.API_URL,
  withCredentials: true, // ✅ CRITICAL: Send cookies with every request
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor for logging
axiosInstance.interceptors.request.use(
  (config) => {
    console.log("🌐 [AXIOS REQUEST]", config.method?.toUpperCase(), config.url);
    console.log("📍 Full URL:", config.baseURL + config.url);
    if (config.data instanceof FormData) {
      console.log("📦 Data: FormData (file upload)");
    } else {
      console.log("📦 Data:", config.data);
    }
    return config;
  },
  (error) => {
    console.error("❌ [AXIOS REQUEST ERROR]", error);
    return Promise.reject(error);
  },
);

// ❌ REMOVED: Authorization header interceptor (cookies are sent automatically)
// No need to manually add tokens to headers

// Response interceptor to handle token refresh
axiosInstance.interceptors.response.use(
  (response) => {
    console.log("✅ [AXIOS RESPONSE]", response.status, response.config.url);
    console.log("📥 Data:", response.data);
    return response;
  },
  async (error) => {
    console.error(
      "❌ [AXIOS ERROR]",
      error.response?.status,
      error.config?.url,
    );
    console.error("📥 Error Data:", error.response?.data);

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

export const apiClient = axiosInstance;
export default axiosInstance;
