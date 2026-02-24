import { useState, useRef, useEffect } from "react";
import * as Comlink from "comlink";

import { GENERATOR_CONFIG, CAT_BREEDS } from "../../constants";
import type { GenerationProgress, GenerationResult } from "../../engine/generation.worker";

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
  Grid,
  Card,
  CardMedia
} from "@mui/material";
import AutoFixHighIcon from "@mui/icons-material/AutoFixHigh";
import RefreshIcon from "@mui/icons-material/Refresh";
import DownloadIcon from "@mui/icons-material/Download";

type GenerationWorker = Comlink.Remote<
  import("../../engine/generation.worker").default
>;

export default function GeneratePage() {
  const [ready, setReady] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<GenerationProgress | null>(null);

  // Generation parameters
  const [breedIndex, setBreedIndex] = useState<number>(0);
  const [steps, setSteps] = useState<number>(GENERATOR_CONFIG.defaultSteps);
  const [cfgScale, setCfgScale] = useState<number>(GENERATOR_CONFIG.defaultCfgScale);
  const [noiseSeed, setNoiseSeed] = useState<number | undefined>(undefined);

  const workerRef = useRef<GenerationWorker | null>(null);

  // Load the generator model on mount
  useEffect(() => {
    const worker = new Worker(
      new URL("../../engine/generation.worker.ts", import.meta.url),
      { type: "module" }
    );

    const engine: GenerationWorker = Comlink.wrap(worker);
    workerRef.current = engine;

    engine.loadModel()
      .then(() => setReady(true))
      .catch((err: unknown) => {
        console.error("Failed to load generator model:", err);
        setError("Failed to load generator model. Please refresh the page.");
      });

    return () => worker.terminate();
  }, []);

  const handleGenerate = async () => {
    if (!workerRef.current || isGenerating) return;

    setError(null);
    setIsGenerating(true);
    setProgress(null);

    try {
      const progressCallback = Comlink.proxy((prog: GenerationProgress) => {
        setProgress(prog);
      });

      const result: GenerationResult = await workerRef.current.generate(
        {
          breedIndex,
          steps,
          cfgScale,
          noiseSeed
        },
        progressCallback
      );

      // Convert ImageData to base64 for display
      const canvas = document.createElement("canvas");
      canvas.width = result.imageData.width;
      canvas.height = result.imageData.height;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.putImageData(result.imageData, 0, 0);
        setGeneratedImage(canvas.toDataURL("image/png"));
      }

      console.log(`Generation complete: ${result.totalTime.toFixed(2)}ms total, ${result.avgStepTime.toFixed(2)}ms/step`);
    } catch (err) {
      console.error("Generation failed:", err);
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setIsGenerating(false);
      setProgress(null);
      // Generate new random seed for next generation
      setNoiseSeed(undefined);
    }
  };

  const handleResetNoise = () => {
    setNoiseSeed(Math.floor(Math.random() * 1000000));
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
        {!ready && (
          <Alert severity="info" sx={{ mt: 2, maxWidth: 900 }}>
            Loading generator model... This may take a moment (126 MB).
          </Alert>
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
            />
            <Typography variant="caption" color="text.secondary">
              Higher = more adherence to breed, lower = more creative
            </Typography>
          </Box>

          {/* Action Buttons */}
          <Button
            variant="contained"
            size="large"
            startIcon={<AutoFixHighIcon />}
            onClick={handleGenerate}
            disabled={isGenerating || !ready}
            fullWidth
            sx={{ mb: 2 }}
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
          {isGenerating && progress && (
            <Box sx={{ width: "100%", mt: 2 }}>
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
