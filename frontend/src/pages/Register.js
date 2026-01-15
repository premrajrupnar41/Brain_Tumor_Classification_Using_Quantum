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
    width: "400px",
    borderRadius: "12px",
    boxShadow: "0 15px 40px rgba(0,0,0,0.2)",
    textAlign: "center",
  },
  input: {
    width: "100%",
    padding: "12px",
    margin: "8px 0",
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

const Register = () => {
  const navigate = useNavigate();

  const [form, setForm] = useState({
    hospital_name: "",
    email: "",
    contact: "",
    name: "",
    address: "",
    username: "",
    password: "",
  });

  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleRegister = async (e) => {
    e.preventDefault();

    for (let key in form) {
      if (!form[key]) {
        alert("Please fill all fields");
        return;
      }
    }

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(form.email)) {
      alert("Please enter a valid email address (e.g., user@example.com)");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8001/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      const data = await response.json();
      console.log("Response:", response.status, data);

      if (response.ok) {
        alert("✓ Registration successful! Your account has been created. Please login now.");
        setForm({
          hospital_name: "",
          email: "",
          contact: "",
          name: "",
          address: "",
          username: "",
          password: "",
        });
        navigate("/");
      } else {
        // Extract error message from validation error
        let errorMsg = "Registration failed";
        if (data.detail) {
          if (typeof data.detail === 'string') {
            errorMsg = data.detail;
          } else if (Array.isArray(data.detail)) {
            // Pydantic validation error - extract the message
            errorMsg = data.detail[0]?.msg || "Invalid input";
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
      <form style={styles.box} onSubmit={handleRegister}>
        <h2>Register</h2>

        <input style={styles.input} name="hospital_name" placeholder="Hospital Name" onChange={handleChange} />
        <input style={styles.input} name="email" placeholder="Email" onChange={handleChange} />
        <input style={styles.input} name="contact" placeholder="Contact" onChange={handleChange} />
        <input style={styles.input} name="name" placeholder="Name" onChange={handleChange} />
        <input style={styles.input} name="address" placeholder="Address" onChange={handleChange} />
        <input style={styles.input} name="username" placeholder="Username" onChange={handleChange} />
        <input style={styles.input} type="password" name="password" placeholder="Password" onChange={handleChange} />

        <button style={styles.button} type="submit" disabled={loading}>
          {loading ? "Registering..." : "Register"}
        </button>

        <p style={{ marginTop: "15px" }}>
          Already have an account?{" "}
          <span style={styles.link} onClick={() => navigate("/")}>
            Login
          </span>
        </p>
      </form>
    </div>
  );
};

export default Register;
