import { Suspense, lazy } from "react";
import { BrowserRouter, Route, Routes } from "react-router";
import LandingPage from "../landingpage/LandingPage";
import SignIn from "../pages/auth/SignIn";
import SignUp from "../pages/auth/SignUp";
import LoadingFallback from "../components/common/LoadingFallback";

// Lazy load non-critical pages
const OTPVerification = lazy(() => import("../pages/auth/OTPVerification"));
const AdminLogin = lazy(() => import("../pages/auth/AdminLogin"));
const StudentDashboard = lazy(() => import("../pages/dashboard/StudentDashboard"));
const AdminDashboard = lazy(() => import("../pages/dashboard/AdminDashboard"));
const StudentDetail = lazy(() => import("../pages/dashboard/StudentDetail"));
const PlaygroundPage = lazy(() => import("../pages/playground/PlayGround"));
const Profile = lazy(() => import("../pages/auth/Profile"));
const ForgotPassword = lazy(() => import("../pages/auth/ForgotPassword"));

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
          <Route path="/dashboard" element={<StudentDashboard />} />
          <Route path="/profile" element={<Profile />} />
          <Route path="/playground" element={<PlaygroundPage />} />
          <Route path="/playground/:projectId" element={<PlaygroundPage />} />
          
          {/* Admin Routes */}
          <Route path="/admin/login" element={<AdminLogin />} />
          <Route path="/admin/dashboard" element={<AdminDashboard />} />
          <Route path="/admin/students/:id" element={<StudentDetail />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
};

export default App;
