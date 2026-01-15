import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Login from "./pages/Login";
import Register from "./pages/Register";
import QML from "./pages/QML";
import CML from "./pages/CML";

function App() {
  // Check if user is logged in by checking localStorage
  const isLoggedIn = localStorage.getItem("username");

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/qml" element={isLoggedIn ? <QML /> : <Navigate to="/" />} />
        <Route path="/cml" element={isLoggedIn ? <CML /> : <Navigate to="/" />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
