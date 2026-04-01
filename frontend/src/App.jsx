import { useState, useCallback } from 'react'
import './App.css'

export default function App() {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)

  const handleFile = (file) => {
    if (!file) return
    if (!['image/jpeg', 'image/png'].includes(file.type)) {
      setError('Only JPG/PNG accepted')
      return
    }
    setImage(file)
    setPreview(URL.createObjectURL(file))
    setResult(null)
    setError(null)
  }

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }, [])

  const handleDetect = async () => {
    if (!image) return
    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', image)

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        body: formData
      })
      if (!res.ok) throw new Error('API error')
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError('Something went wrong. Is the API running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <header>
        <h1>DEEPFAKE<span>DETECTOR</span></h1>
        <p className="subtitle">CNN + RAG powered forensic analysis</p>
      </header>

      <div
        className={`dropzone ${dragging ? 'dragging' : ''} ${preview ? 'has-image' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => document.getElementById('fileInput').click()}
      >
        {preview
          ? <img src={preview} alt="preview" className="preview" />
          : <div className="dropzone-text">
              <span className="icon">⬆</span>
              <p>Drop image here or click to upload</p>
              <p className="hint">JPG, PNG supported</p>
            </div>
        }
        <input
          id="fileInput"
          type="file"
          accept=".jpg,.jpeg,.png"
          style={{ display: 'none' }}
          onChange={(e) => handleFile(e.target.files[0])}
        />
      </div>

      {error && <p className="error">{error}</p>}

      <button
        className="detect-btn"
        onClick={handleDetect}
        disabled={!image || loading}
      >
        {loading ? 'ANALYSING...' : 'DETECT'}
      </button>

      {result && (
        <div className={`result-card ${result.label}`}>
          <div className="result-header">
            <span className="label">{result.label.toUpperCase()}</span>
            <span className="confidence">{result.confidence}%</span>
          </div>
          <div className="confidence-bar">
            <div className="fill" style={{ width: `${result.confidence}%` }} />
          </div>
          <div className="explanation">
            <p className="explanation-title">ANALYSIS</p>
            <p>{result.explanation}</p>
          </div>
        </div>
      )}
    </div>
  )
}