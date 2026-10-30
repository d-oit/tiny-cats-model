import { useState } from "react";

import { GENERATOR_CONFIG, CAT_BREEDS } from "../../constants";
import { useGeneration } from "../../hooks/useGeneration";
import type { GenerationProgress } from "../../engine/generation.worker";
import { imageDataToCanvas } from "../../utils/imageUtils";

import {
  Typography,
  Box,
  Button,
  Paper,
  LinearProgress,
  Alert,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Grid,
  Card,
  CardMedia
} from "@mui/material";
import AutoFixHighIcon from "@mui/icons-material/AutoFixHigh";
import RefreshIcon from "@mui/icons-material/Refresh";
import DownloadIcon from "@mui/icons-material/Download";

export default function GeneratePage() {
  const {
    ready,
    status,
    error: modelError,
    provider,
    source,
    retry,
    generate,
  } = useGeneration();
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<GenerationProgress | null>(null);

  // Generation parameters
  const [breedIndex, setBreedIndex] = useState<number>(0);
  const [steps, setSteps] = useState<number>(GENERATOR_CONFIG.defaultSteps);
  const [cfgScale, setCfgScale] = useState<number>(GENERATOR_CONFIG.defaultCfgScale);
  const [seedInput, setSeedInput] = useState<string>("");

  const parsedSeed = (): number | undefined => {
    if (seedInput.trim() === "") return undefined;
    const value = Number(seedInput);
    if (!Number.isInteger(value) || value < 0) {
      throw new Error("Noise seed must be a non-negative integer (or blank for random).");
    }
    return value;
  };

  const handleGenerate = async () => {
    if (!ready || isGenerating) return;

    setError(null);
    setProgress(null);

    let noiseSeed: number | undefined;
    try {
      noiseSeed = parsedSeed();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Invalid noise seed");
      return;
    }

    setIsGenerating(true);
    try {
      const result = await generate(
        { breedIndex, steps, cfgScale, noiseSeed },
        (update) => setProgress(update)
      );

      const canvas = imageDataToCanvas(
        result.imageData,
        result.imageData.width,
        result.imageData.height
      );
      setGeneratedImage(canvas.toDataURL("image/png"));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setIsGenerating(false);
      setProgress(null);
    }
  };

  const handleResetNoise = () => {
    setSeedInput(String(Math.floor(Math.random() * 1000000)));
    setGeneratedImage(null);
    setProgress(null);
    setError(null);
  };

  const handleDownload = () => {
    if (!generatedImage) return;

    const link = document.createElement("a");
    link.href = generatedImage;
    link.download = `cat_${CAT_BREEDS[breedIndex]?.name.toLowerCase().replace(/\s+/g, "_")}_${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const breedName = CAT_BREEDS.find(b => b.index === breedIndex)?.name || "Random";

  return (
    <>
      <div className="column container">
        <Typography variant="h3" fontWeight="400" sx={{ mt: 1, mb: 2, letterSpacing: "-0.02em" }}>
          Cat Image Generator
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
          Generate cat images using the TinyDiT model with breed conditioning.
          Running entirely locally in your browser using ONNX Runtime Web.
        </Typography>
        {status === "loading" && (
          <Alert severity="info" sx={{ mt: 2, maxWidth: 900 }}>
            Loading and verifying the generator model... This may take a moment (126 MB).
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
        {/* Control Panel */}
        <Paper
          elevation={3}
          sx={{
            p: 3,
            display: "flex",
            flexDirection: "column",
            minWidth: 350,
            maxWidth: 450
          }}
        >
          <Typography variant="h6" gutterBottom>
            Generation Settings
          </Typography>

          {/* Breed Selector */}
          <FormControl fullWidth sx={{ mt: 2, mb: 3 }}>
            <InputLabel id="breed-select-label">Cat Breed</InputLabel>
            <Select
              labelId="breed-select-label"
              value={breedIndex}
              label="Cat Breed"
              onChange={(e) => setBreedIndex(Number(e.target.value))}
              disabled={isGenerating || !ready}
              inputProps={{ "data-testid": "breed-select" }}
            >
              {CAT_BREEDS.map((breed) => (
                <MenuItem key={breed.index} value={breed.index}>
                  {breed.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Steps Slider */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              Sampling Steps: {steps}
            </Typography>
            <Slider
              value={steps}
              onChange={(_, value) => setSteps(Number(value))}
              min={GENERATOR_CONFIG.minSteps}
              max={GENERATOR_CONFIG.maxSteps}
              step={5}
              disabled={isGenerating || !ready}
              valueLabelDisplay="auto"
              aria-label="Sampling steps"
            />
            <Typography variant="caption" color="text.secondary">
              More steps = better quality but slower
            </Typography>
          </Box>

          {/* CFG Scale Slider */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              Guidance Scale (CFG): {cfgScale.toFixed(1)}
            </Typography>
            <Slider
              value={cfgScale}
              onChange={(_, value) => setCfgScale(Number(value))}
              min={GENERATOR_CONFIG.minCfgScale}
              max={GENERATOR_CONFIG.maxCfgScale}
              step={0.1}
              disabled={isGenerating || !ready}
              valueLabelDisplay="auto"
              aria-label="Guidance scale"
            />
            <Typography variant="caption" color="text.secondary">
              Higher = more adherence to breed, lower = more creative
            </Typography>
          </Box>

          {/* Deterministic seed */}
          <TextField
            label="Noise seed (blank = random)"
            type="number"
            value={seedInput}
            onChange={(event) => setSeedInput(event.target.value)}
            disabled={isGenerating || !ready}
            inputProps={{ min: 0, step: 1, "data-testid": "noise-seed" }}
            size="small"
            fullWidth
            sx={{ mb: 3 }}
          />

          {/* Action Buttons */}
          <Button
            variant="contained"
            size="large"
            startIcon={<AutoFixHighIcon />}
            onClick={handleGenerate}
            disabled={isGenerating || !ready}
            fullWidth
            sx={{ mb: 2 }}
            data-testid="generate-button"
          >
            {isGenerating ? "Generating..." : "Generate"}
          </Button>

          <Grid container spacing={2}>
            <Grid size={{ xs: 6 }}>
              <Button
                variant="outlined"
                startIcon={<RefreshIcon />}
                onClick={handleResetNoise}
                disabled={isGenerating || !ready}
                fullWidth
              >
                Reset Noise
              </Button>
            </Grid>
            <Grid size={{ xs: 6 }}>
              <Button
                variant="outlined"
                startIcon={<DownloadIcon />}
                onClick={handleDownload}
                disabled={!generatedImage}
                fullWidth
              >
                Download
              </Button>
            </Grid>
          </Grid>

          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
        </Paper>

        {/* Generation Canvas */}
        <Paper
          elevation={3}
          sx={{
            p: 3,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            minWidth: 400,
            minHeight: 450
          }}
        >
          <Typography variant="h6" gutterBottom>
            Generated Image
          </Typography>

          <Card
            sx={{
              width: 256,
              height: 256,
              mt: 2,
              mb: 2,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              backgroundColor: "grey.100",
              border: generatedImage ? "none" : "2px dashed",
              borderColor: "grey.400"
            }}
          >
            {generatedImage ? (
              <CardMedia
                component="img"
                image={generatedImage}
                alt={`Generated ${breedName} cat`}
                sx={{ width: 256, height: 256, objectFit: "cover" }}
                data-testid="generated-image"
              />
            ) : (
              <Typography variant="body2" color="text.secondary" align="center">
                {isGenerating ? "Generating..." : "Click Generate to create an image"}
              </Typography>
            )}
          </Card>

          {generatedImage && (
            <Typography variant="caption" color="text.secondary" sx={{ mb: 2 }}>
              Breed: {breedName}
            </Typography>
          )}

          {/* Progress Dashboard */}
          <Box aria-live="polite" sx={{ width: "100%" }}>
            {isGenerating && progress && (
              <Box sx={{ width: "100%", mt: 2 }} data-testid="generation-progress">
                <LinearProgress
                  variant="determinate"
                  value={(progress.step / progress.totalSteps) * 100}
                />
                <Grid container spacing={2} sx={{ mt: 2 }}>
                  <Grid size={{ xs: 4 }}>
                    <Box sx={{ textAlign: "center" }}>
                      <Typography variant="h6" color="primary">
                        {progress.step}/{progress.totalSteps}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Steps
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid size={{ xs: 4 }}>
                    <Box sx={{ textAlign: "center" }}>
                      <Typography variant="h6" color="primary">
                        {progress.stepTime.toFixed(0)}ms
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Step Time
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid size={{ xs: 4 }}>
                    <Box sx={{ textAlign: "center" }}>
                      <Typography variant="h6" color="primary">
                        {(progress.totalTime / 1000).toFixed(1)}s
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Total
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Box>
            )}
          </Box>

          {!isGenerating && !generatedImage && (
            <Box sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              color: "text.secondary",
              mt: 2
            }}>
              <Typography>
                Adjust settings and click Generate
              </Typography>
            </Box>
          )}
        </Paper>
      </div>
    </>
  );
}