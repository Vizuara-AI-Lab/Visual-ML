import type { ReactNode } from "react";
import { Navigate } from "react-router";

interface ProtectedRouteProps {
  children: ReactNode;
  requireAdmin?: boolean;
}

/**
 * ProtectedRoute component that checks authentication before rendering children
 * Redirects to signin if user is not authenticated
 * Redirects to dashboard if admin route is accessed by non-admin
 */
const ProtectedRoute = ({ children, requireAdmin = false }: ProtectedRouteProps) => {
  // Check if user is authenticated
  const user = localStorage.getItem("user");
  const userRole = localStorage.getItem("userRole");
  
  // If no user data, redirect to signin
  if (!user) {
    return <Navigate to="/signin" replace />;
  }

  // If admin route is required but user is not admin, redirect to dashboard
  if (requireAdmin && userRole !== "ADMIN") {
    return <Navigate to="/dashboard" replace />;
  }

  // User is authenticated, render the protected content
  return <>{children}</>;
};

export default ProtectedRoute;
