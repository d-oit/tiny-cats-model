import { useEffect, useRef, useState } from "react";

import { useInference } from "../../hooks/useInference";
import {
  decodeImageFile,
  preprocessImageSource,
  createPreviewUrl,
  revokePreviewUrl,
  DEFAULT_IMAGE_LIMITS,
  formatBytes,
} from "../../utils/imageUtils";
import { MODEL_CONFIGS, type ModelType } from "../../constants";
import type { ClassificationTelemetry } from "../../types";

import { Typography, Box, Button, Paper, LinearProgress, Alert } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import RefreshIcon from "@mui/icons-material/Refresh";

const MODEL_TYPE: ModelType = "cats";

function friendlyLabel(className: string): string {
  if (className === "cat") return "Cat";
  if (className === "not_cat") return "Not a Cat";
  return className.replace(/_/g, " ");
}

export default function ClassifyPage() {
  const CONFIGS = MODEL_CONFIGS[MODEL_TYPE];

  const fileInputRef = useRef<HTMLInputElement>(null);
  const previewRef = useRef<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<ClassificationTelemetry | null>(null);
  const [error, setError] = useState<string | null>(null);

  const { ready, status, error: modelError, provider, source, retry, classify } =
    useInference(CONFIGS.asset.localPath, MODEL_TYPE);

  useEffect(() => {
    return () => revokePreviewUrl(previewRef.current);
  }, []);

  const resetPreview = () => {
    revokePreviewUrl(previewRef.current);
    previewRef.current = null;
    setPreviewUrl(null);
  };

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setError(null);
    setResult(null);
    resetPreview();

    if (!ready) {
      setError("The model is still loading. Wait for it to be ready, then retry.");
      return;
    }

    let closeDecoded: (() => void) | null = null;
    try {
      setIsProcessing(true);
      const decoded = await decodeImageFile(file);
      closeDecoded = decoded.close;

      const url = await createPreviewUrl(
        decoded.source,
        decoded.width,
        decoded.height
      );
      previewRef.current = url;
      setPreviewUrl(url);

      const tensor = preprocessImageSource(
        decoded.source,
        decoded.width,
        decoded.height,
        CONFIGS.imgDims[0],
        CONFIGS.imgDims[1],
        CONFIGS.mean,
        CONFIGS.std
      );
      if (import.meta.env.DEV) {
        const capture = (
          globalThis as typeof globalThis & {
            __TINY_CATS_CAPTURE_PREPROCESSING__?: (
              value: Float32Array,
              dimensions: { width: number; height: number }
            ) => void;
          }
        ).__TINY_CATS_CAPTURE_PREPROCESSING__;
        capture?.(tensor, { width: decoded.width, height: decoded.height });
      }
      setResult(await classify(tensor));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      closeDecoded?.();
      setIsProcessing(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleUploadClick = () => {
    if (ready && !isProcessing) fileInputRef.current?.click();
  };

  const handleReset = () => {
    resetPreview();
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
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
        {status === "loading" && (
          <Alert severity="info" sx={{ mt: 2, maxWidth: 900 }}>
            Loading and verifying the model. Please wait.
          </Alert>
        )}
        {modelError && (
          <Alert
            severity="error"
            sx={{ mt: 2, maxWidth: 900 }}
            action={
              <Button color="inherit" size="small" onClick={retry}>
                Retry
              </Button>
            }
          >
            {modelError}
          </Alert>
        )}
        {ready && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, pl: 1 }}>
            Verified model · {source} · {provider?.toUpperCase()}
          </Typography>
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
            accept="image/jpeg,image/png,image/webp,image/gif,image/bmp"
            ref={fileInputRef}
            onChange={handleFileSelect}
            aria-label="Choose an image to classify"
            disabled={!ready || isProcessing}
            style={{ display: "none" }}
            data-testid="classify-file-input"
          />

          {previewUrl ? (
            <Box sx={{ position: "relative" }}>
              <img
                src={previewUrl}
                alt="Preview of the uploaded image"
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
              role="button"
              tabIndex={0}
              aria-label="Click to upload an image"
              aria-disabled={!ready || isProcessing}
              data-testid="classify-dropzone"
              onClick={handleUploadClick}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  handleUploadClick();
                }
              }}
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
                cursor: ready ? "pointer" : "not-allowed",
                backgroundColor: "grey.50",
                transition: "all 0.2s",
                "&:hover": {
                  borderColor: ready ? "primary.main" : "grey.400",
                  backgroundColor: ready ? "grey.100" : "grey.50"
                }
              }}
            >
              <CloudUploadIcon sx={{ fontSize: 64, color: "grey.400", mb: 2 }} />
              <Typography variant="body1" color="text.secondary">
                Click to upload an image
              </Typography>
              <Typography variant="caption" color="text.secondary">
                (JPG, PNG, WebP, GIF or BMP, max {formatBytes(DEFAULT_IMAGE_LIMITS.maxFileBytes)})
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

          <Box aria-live="polite">
            {error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}

            {result && !error && (
              <Box sx={{ mt: 2 }} data-testid="classification-result">
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
                    <Box key={CONFIGS.classNames[idx] ?? idx} sx={{ mb: 1 }}>
                      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
                        <Typography variant="body2">
                          {friendlyLabel(CONFIGS.classNames[idx])}
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
          </Box>

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