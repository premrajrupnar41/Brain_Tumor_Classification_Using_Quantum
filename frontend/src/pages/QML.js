import React, { useState } from "react";
import "./QML.css";

function QML() {
  const [image, setImage] = useState(null);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(URL.createObjectURL(file));
    }
  };

  const clearImage = () => {
    setImage(null);
  };

  return (
    <div className="qml-container">
      <div className="qml-card">
        <h1>Quantum ML Disease Classification</h1>
        <p className="subtitle">
          AI-powered medical diagnosis using Quantum Machine Learning
        </p>

        {/* Upload / Change Image */}
        <label className="upload-box">
          {image ? "Change Image" : "Upload Medical Image"}
          <input type="file" accept="image/*" hidden onChange={handleImageUpload} />
        </label>

        {/* Image Preview */}
        {image && (
          <>
            <div className="image-preview">
              <img src={image} alt="Preview" />
            </div>

            {/* Clear Image Button */}
            <button className="clear-btn" onClick={clearImage}>
              Clear Image
            </button>
          </>
        )}

        {/* Analyze Button */}
        <button className="classify-btn">
          Analyze Disease
        </button>

        <p className="footer-text">
          Secure • Fast • Research-Driven
        </p>
      </div>
    </div>
  );
}

export default QML;
