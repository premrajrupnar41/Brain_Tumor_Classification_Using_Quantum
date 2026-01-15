import { BrowserRouter, Routes, Route } from "react-router-dom";
import Login from "./pages/Login";
import Register from "./pages/Register";
import QML from "./pages/QML";
import CML from "./pages/CML";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/qml" element={<QML />} />
        <Route path="/cml" element={<CML />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
