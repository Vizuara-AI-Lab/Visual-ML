import { Suspense, lazy } from "react";
import { BrowserRouter, Route, Routes } from "react-router";
import LandingPage from "../landingpage/LandingPage";
import SignIn from "../pages/auth/SignIn";
import SignUp from "../pages/auth/SignUp";
import LoadingFallback from "../components/common/LoadingFallback";
import NotFound from "../pages/common/NotFound";

// Lazy load non-critical pages
const OTPVerification = lazy(() => import("../pages/auth/OTPVerification"));
const AdminLogin = lazy(() => import("../pages/auth/AdminLogin"));
const StudentDashboard = lazy(
  () => import("../pages/dashboard/StudentDashboard"),
);
const AdminDashboard = lazy(() => import("../pages/dashboard/AdminDashboard"));
const StudentDetail = lazy(() => import("../pages/dashboard/StudentDetail"));
const PlaygroundPage = lazy(() => import("../pages/playground/PlayGround"));
const Profile = lazy(() => import("../pages/auth/Profile"));
const ForgotPassword = lazy(() => import("../pages/auth/ForgotPassword"));
const SharedProject = lazy(() => import("../pages/SharedProject"));
const ProtectedRoute = lazy(
  () => import("../components/common/ProtectedRoute"),
);

const App = () => {
  return (
    <BrowserRouter>
      <Suspense fallback={<LoadingFallback />}>
        <Routes>
          {/* Eager loaded routes */}
          <Route path="/" element={<LandingPage />} />
          <Route path="/signin" element={<SignIn />} />
          <Route path="/signup" element={<SignUp />} />

          {/* Lazy loaded routes */}
          <Route path="/verify-email" element={<OTPVerification />} />
          <Route path="/forgot-password" element={<ForgotPassword />} />

          {/* Public shared project view */}
          <Route path="/shared/:shareToken" element={<SharedProject />} />

          <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <StudentDashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/profile"
            element={
              <ProtectedRoute>
                <Profile />
              </ProtectedRoute>
            }
          />
          <Route
            path="/playground"
            element={
              <ProtectedRoute>
                <PlaygroundPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/playground/:projectId"
            element={
              <ProtectedRoute>
                <PlaygroundPage />
              </ProtectedRoute>
            }
          />

          {/* Admin Routes */}
          <Route path="/admin/login" element={<AdminLogin />} />
          <Route
            path="/admin/dashboard"
            element={
              <ProtectedRoute requireAdmin>
                <AdminDashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/admin/students/:id"
            element={
              <ProtectedRoute requireAdmin>
                <StudentDetail />
              </ProtectedRoute>
            }
          />

          {/* 404 Catch-all route - must be last */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
};

export default App;
