import { useState, useRef } from "react";

import { useInference } from "../../hooks/useInference";
import { loadImageAsTensor, centerCropAndResize } from "../../utils/imageUtils";
import { MODEL_CONFIGS } from "../../constants";
import type { ClassificationTelemetry } from "../../types";

import { Typography, Box, Button, Paper, LinearProgress, Alert } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import RefreshIcon from "@mui/icons-material/Refresh";

export default function ClassifyPage() {
  const MODEL_TYPE = "cats";
  const CONFIGS = MODEL_CONFIGS[MODEL_TYPE];
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<ClassificationTelemetry | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const { ready, classify } = useInference(CONFIGS.modelPath, MODEL_TYPE);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setError(null);
    setResult(null);

    const reader = new FileReader();
    reader.onload = async (event) => {
      const base64 = event.target?.result as string;
      setPreviewUrl(base64);
      
      try {
        setIsProcessing(true);
        
        const processedImage = await centerCropAndResize(base64, CONFIGS.imgDims[0], CONFIGS.imgDims[1]);
        const tensor = await loadImageAsTensor(processedImage, MODEL_TYPE);
        
        const classificationResult = await classify(tensor);
        
        if (classificationResult) {
          setResult(classificationResult);
        } else {
          setError("Classification failed");
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        setIsProcessing(false);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleReset = () => {
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <>
      <div className="column container">
        <Typography variant="h3" fontWeight="400" sx={{ mt: 1, mb: 2, letterSpacing: "-0.02em" }}>
          Cat Classifier
        </Typography>
        <Typography variant="body1" component="div"
          sx={{
            fontSize: "1.125rem",
            lineHeight: 1.75,
            textAlign: "left",
            paddingLeft: "1em",
            paddingRight: "1em",
            maxWidth: "1100px"
          }}
        >
          Upload an image to classify whether it contains a cat or not. 
          Running entirely locally in your browser using ONNX Runtime Web.
        </Typography>
        {!ready && (
          <Alert severity="info" sx={{ mt: 2, maxWidth: 900 }}>
            Loading model... Please wait.
          </Alert>
        )}
      </div>

      <div className="row container" style={{ justifyContent: "center", gap: "2rem" }}>
        <Paper 
          elevation={3}
          sx={{ 
            p: 3, 
            display: "flex", 
            flexDirection: "column", 
            alignItems: "center",
            minWidth: 300,
            minHeight: 400
          }}
        >
          <input
            type="file"
            accept="image/*"
            ref={fileInputRef}
            onChange={handleFileSelect}
            style={{ display: "none" }}
          />

          {previewUrl ? (
            <Box sx={{ position: "relative" }}>
              <img 
                src={previewUrl} 
                alt="Preview" 
                style={{ 
                  maxWidth: 300, 
                  maxHeight: 300, 
                  borderRadius: 8,
                  objectFit: "cover"
                }} 
              />
              <Button
                variant="contained"
                startIcon={<RefreshIcon />}
                onClick={handleReset}
                sx={{ mt: 2 }}
                fullWidth
              >
                Try Another Image
              </Button>
            </Box>
          ) : (
            <Box 
              sx={{ 
                width: 300, 
                height: 300, 
                display: "flex", 
                flexDirection: "column",
                alignItems: "center", 
                justifyContent: "center",
                border: "2px dashed",
                borderColor: "grey.400",
                borderRadius: 2,
                cursor: "pointer",
                backgroundColor: "grey.50",
                transition: "all 0.2s",
                "&:hover": {
                  borderColor: "primary.main",
                  backgroundColor: "grey.100"
                }
              }}
              onClick={handleUploadClick}
            >
              <CloudUploadIcon sx={{ fontSize: 64, color: "grey.400", mb: 2 }} />
              <Typography variant="body1" color="text.secondary">
                Click to upload an image
              </Typography>
              <Typography variant="caption" color="text.secondary">
                (JPG, PNG, etc.)
              </Typography>
            </Box>
          )}

          {isProcessing && (
            <Box sx={{ width: "100%", mt: 2 }}>
              <LinearProgress />
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: "block" }}>
                Processing image...
              </Typography>
            </Box>
          )}
        </Paper>

        <Paper 
          elevation={3}
          sx={{ 
            p: 3, 
            display: "flex", 
            flexDirection: "column", 
            minWidth: 300,
            minHeight: 400
          }}
        >
          <Typography variant="h6" gutterBottom>
            Classification Result
          </Typography>

          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}

          {result && !error && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="h4" sx={{ color: "primary.main", fontWeight: "bold" }}>
                {result.result.className === "cat" ? "🐱 Cat" : "🚫 Not a Cat"}
              </Typography>
              
              <Typography variant="h6" sx={{ mt: 2 }}>
                Confidence: {(result.result.confidence * 100).toFixed(1)}%
              </Typography>

              <Box sx={{ mt: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Class Probabilities:
                </Typography>
                {result.probabilities.map((prob, idx) => (
                  <Box key={idx} sx={{ mb: 1 }}>
                    <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
                      <Typography variant="body2">
                        {CONFIGS.classNames[idx]}
                      </Typography>
                      <Typography variant="body2">
                        {(prob * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={prob * 100} 
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                  </Box>
                ))}
              </Box>
            </Box>
          )}

          {!result && !error && (
            <Box sx={{ 
              display: "flex", 
              alignItems: "center", 
              justifyContent: "center", 
              height: "100%",
              color: "text.secondary"
            }}>
              <Typography>
                Upload an image to see classification results
              </Typography>
            </Box>
          )}
        </Paper>
      </div>
    </>
  );
}
