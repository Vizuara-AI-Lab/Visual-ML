import { BrowserRouter, Route, Routes } from "react-router";
import LandingPage from "../landingpage/LandingPage";
import SignIn from "../pages/auth/SignIn";
import SignUp from "../pages/auth/SignUp";
import StudentDashboard from "../pages/dashboard/StudentDashboard";
import PlaygroundPage from "../pages/playground/PlayGround";

const App = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/signin" element={<SignIn />} />
        <Route path="/signup" element={<SignUp />} />
        <Route path="/dashboard" element={<StudentDashboard />} />
        <Route path="/playground" element={<PlaygroundPage />} />
      </Routes>
    </BrowserRouter>
  );
};

export default App;
