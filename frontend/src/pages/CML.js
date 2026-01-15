import React, { useState } from "react";
import "./CML.css";

function CML() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImage = (e) => {
  const file = e.target.files && e.target.files[0];

  // 🔥 VERY IMPORTANT CHECK
  if (!file) {
    return; // user clicked Cancel → do nothing
  }

  setImage(file);
  setPreview(URL.createObjectURL(file));
  setResult(null);
};

  const predictDisease = async () => {
    if (!image) {
      alert("Please upload MRI image");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    setLoading(true);

    const response = await fetch("http://127.0.0.1:8001/predict-cnn", {
      method: "POST",
      body: formData,
    });
    let data;
    try {
      data = await response.json();
    } catch (e) {
      data = { error: 'Invalid JSON response from server' };
    }
    if (!response.ok) {
      setResult({ error: data.error || 'Server returned an error' });
    } else {
      setResult(data);
    }
    setLoading(false);
  };

  const clearAll = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
  };

  return (
    <div className="cml-container">
      <h1>Brain Tumor Classification (CNN)</h1>

      <input type="file" accept="image/*" onChange={handleImage} />

      {preview && (
        <div className="image-box">
          <img src={preview} alt="MRI Preview" />
        </div>
      )}

      <div className="btn-group">
        <button onClick={predictDisease}>Predict</button>
        <button className="clear" onClick={clearAll}>Clear</button>
      </div>

      {loading && <p>Analyzing MRI...</p>}

      {result && result.error && (
        <div className="result-box error">
          <h2>Error</h2>
          <p>{result.error}</p>
        </div>
      )}

      {result && !result.error && (
        <div className="result-box">
          <h2> Tumor Type: {result.tumor_type}</h2>
          <p>Confidence: {result.confidence}%</p>
        </div>
      )}
    </div>
  );
}

export default CML;
