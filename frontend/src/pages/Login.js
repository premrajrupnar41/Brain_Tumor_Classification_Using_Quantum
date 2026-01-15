import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const styles = {
  container: {
    height: "100vh",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    background: "linear-gradient(135deg, #4e54c8, #8f94fb)",
  },
  box: {
    background: "#fff",
    padding: "35px",
    width: "360px",
    borderRadius: "12px",
    boxShadow: "0 15px 40px rgba(0,0,0,0.2)",
    textAlign: "center",
  },
  input: {
    width: "100%",
    padding: "12px",
    margin: "10px 0",
    borderRadius: "6px",
    border: "1px solid #ccc",
    fontSize: "14px",
  },
  button: {
    width: "100%",
    padding: "12px",
    background: "#4e54c8",
    color: "#fff",
    border: "none",
    borderRadius: "6px",
    marginTop: "15px",
    cursor: "pointer",
    fontSize: "16px",
  },
  link: {
    color: "#4e54c8",
    cursor: "pointer",
    fontWeight: "bold",
  },
};

const Login = () => {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();

    if (!username || !password) {
      alert("Please fill all fields");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8001/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      const data = await response.json();
      console.log("Login Response:", response.status, data);

      if (response.ok) {
        localStorage.setItem("username", username);
        localStorage.setItem("hospital", data.hospital || "");
        alert("✓ Login successful! Welcome to Brain Tumor Classification.");
        // Delay navigation slightly to ensure localStorage is set before page loads
        setTimeout(() => {
          navigate("/qml", { replace: true });
        }, 500);
      } else {
        let errorMsg = "Login failed";
        if (data.detail) {
          if (typeof data.detail === 'string') {
            errorMsg = data.detail;
          } else {
            errorMsg = JSON.stringify(data.detail);
          }
        }
        alert("✗ " + errorMsg);
      }
    } catch (error) {
      console.error("Error:", error);
      alert("✗ Error: " + (error?.message || "Network error"));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <form style={styles.box} onSubmit={handleLogin}>
        <h2>Login</h2>

        <input
          style={styles.input}
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />

        <input
          style={styles.input}
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        <button style={styles.button} type="submit" disabled={loading}>
          {loading ? "Logging in..." : "Login"}
        </button>

        <p style={{ marginTop: "15px" }}>
          Don’t have an account?{" "}
          <span style={styles.link} onClick={() => navigate("/register")}>
            Register
          </span>
        </p>
      </form>
    </div>
  );
};

export default Login;
