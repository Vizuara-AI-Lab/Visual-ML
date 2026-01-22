import { BrowserRouter, Route, Routes } from "react-router";
import LandingPage from "../landingpage/LandingPage";
import SignIn from "../pages/auth/SignIn";
import SignUp from "../pages/auth/SignUp";
import OTPVerification from "../pages/auth/OTPVerification";
import AdminLogin from "../pages/auth/AdminLogin";
import StudentDashboard from "../pages/dashboard/StudentDashboard";
import AdminDashboard from "../pages/dashboard/AdminDashboard";
import StudentDetail from "../pages/dashboard/StudentDetail";
import PlaygroundPage from "../pages/playground/PlayGround";
import Profile from "../pages/auth/Profile";

const App = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/signin" element={<SignIn />} />
        <Route path="/signup" element={<SignUp />} />
        <Route path="/verify-email" element={<OTPVerification />} />
        <Route path="/dashboard" element={<StudentDashboard />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/playground" element={<PlaygroundPage />} />
        <Route path="/playground/:projectId" element={<PlaygroundPage />} />
        
        {/* Admin Routes */}
        <Route path="/admin/login" element={<AdminLogin />} />
        <Route path="/admin/dashboard" element={<AdminDashboard />} />
        <Route path="/admin/students/:id" element={<StudentDetail />} />
      </Routes>
    </BrowserRouter>
  );
};

export default App;
