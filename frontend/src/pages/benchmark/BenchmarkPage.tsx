import { useState, useCallback } from "react";
import {
  Typography,
  Box,
  Button,
  Paper,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Chip,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  IconButton,
  Tooltip,
} from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import RefreshIcon from "@mui/icons-material/Refresh";
import DownloadIcon from "@mui/icons-material/Download";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";
import InfoIcon from "@mui/icons-material/Info";

import {
  runBenchmarkSuite,
  formatLatency,
  getPerformanceRecommendations,
  exportReportToMarkdown,
  type BenchmarkReport,
  type BenchmarkStats,
} from "../utils/benchmark";

const GOAP_GOAL_THRESHOLD = 2000; // 2 seconds in ms

function StatsRow({ label, stats }: { label: string; stats: BenchmarkStats }) {
  return (
    <TableRow>
      <TableCell>{label}</TableCell>
      <TableCell align="right">{formatLatency(stats.mean)}</TableCell>
      <TableCell align="right">{formatLatency(stats.std)}</TableCell>
      <TableCell align="right">{formatLatency(stats.min)}</TableCell>
      <TableCell align="right">{formatLatency(stats.max)}</TableCell>
      <TableCell align="right">{formatLatency(stats.p50)}</TableCell>
      <TableCell align="right">{formatLatency(stats.p95)}</TableCell>
      <TableCell align="right">{formatLatency(stats.p99)}</TableCell>
    </TableRow>
  );
}

export default function BenchmarkPage() {
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState("");
  const [report, setReport] = useState<BenchmarkReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRunBenchmark = useCallback(async () => {
    setIsRunning(true);
    setError(null);
    setReport(null);
    setProgress("Initializing benchmark...");

    try {
      const result = await runBenchmarkSuite(setProgress);
      setReport(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Benchmark failed");
    } finally {
      setIsRunning(false);
    }
  }, []);

  const handleDownloadReport = useCallback(() => {
    if (!report) return;

    const markdown = exportReportToMarkdown(report);
    const blob = new Blob([markdown], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `benchmark-report-${Date.now()}.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [report]);

  const getGoalStatus = () => {
    if (!report) return null;

    const meetsGoal = report.summary.meetsGoal;
    return (
      <Chip
        icon={meetsGoal ? <CheckCircleIcon /> : <ErrorIcon />}
        label={meetsGoal ? "GOAL MET (<2s)" : "GOAL NOT MET"}
        color={meetsGoal ? "success" : "error"}
        sx={{ fontWeight: "bold" }}
      />
    );
  };

  return (
    <>
      <div className="column container">
        <Typography variant="h3" fontWeight="400" sx={{ mt: 1, mb: 2, letterSpacing: "-0.02em" }}>
          Performance Benchmark
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
          Benchmark classification and generation inference latency in your browser.
          Results are compared against GOAP success metrics (&lt;2s for full generation).
        </Typography>

        <Box sx={{ mt: 3, mb: 3, display: "flex", gap: 2, alignItems: "center" }}>
          <Button
            variant="contained"
            size="large"
            startIcon={isRunning ? <CircularProgress size={20} /> : <PlayArrowIcon />}
            onClick={handleRunBenchmark}
            disabled={isRunning}
          >
            {isRunning ? "Running..." : "Run Benchmark"}
          </Button>

          {report && (
            <>
              <Button
                variant="outlined"
                startIcon={<RefreshIcon />}
                onClick={handleRunBenchmark}
                disabled={isRunning}
              >
                Re-run
              </Button>
              <Button
                variant="outlined"
                startIcon={<DownloadIcon />}
                onClick={handleDownloadReport}
              >
                Download Report
              </Button>
              {getGoalStatus()}
            </>
          )}
        </Box>

        {isRunning && (
          <Alert severity="info" sx={{ mb: 2 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
              <CircularProgress size={20} />
              <Typography variant="body2">{progress}</Typography>
            </Box>
            <LinearProgress sx={{ mt: 2 }} />
          </Alert>
        )}

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
      </div>

      {report && (
        <div className="row container" style={{ justifyContent: "center", gap: "2rem", flexDirection: "column" }}>
          {/* System Info */}
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              System Information
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6} md={3}>
                <Typography variant="caption" color="text.secondary">CPU Cores</Typography>
                <Typography variant="body1">{report.hardwareConcurrency}</Typography>
              </Grid>
              <Grid item xs={6} md={3}>
                <Typography variant="caption" color="text.secondary">Browser</Typography>
                <Typography variant="body2" sx={{ fontSize: "0.75rem" }}>
                  {report.userAgent.split(" ").slice(0, 3).join(" ")}...
                </Typography>
              </Grid>
              <Grid item xs={6} md={3}>
                <Typography variant="caption" color="text.secondary">Timestamp</Typography>
                <Typography variant="body2">
                  {new Date(report.timestamp).toLocaleString()}
                </Typography>
              </Grid>
              <Grid item xs={6} md={3}>
                <Typography variant="caption" color="text.secondary">GOAP Goal</Typography>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  {report.summary.meetsGoal ? (
                    <CheckCircleIcon color="success" fontSize="small" />
                  ) : (
                    <ErrorIcon color="error" fontSize="small" />
                  )}
                  <Typography variant="body2">
                    {report.summary.meetsGoal ? "Passed" : "Failed"}
                  </Typography>
                </Box>
              </Grid>
            </Grid>
          </Paper>

          {/* Classification Results */}
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Classification Latency
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: "block" }}>
              Measured across different image sizes (10 runs each)
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Image Size</TableCell>
                    <TableCell align="right">Mean</TableCell>
                    <TableCell align="right">Std</TableCell>
                    <TableCell align="right">Min</TableCell>
                    <TableCell align="right">Max</TableCell>
                    <TableCell align="right">P50</TableCell>
                    <TableCell align="right">P95</TableCell>
                    <TableCell align="right">P99</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {report.classification.map((result) => (
                    <StatsRow
                      key={result.imageSize}
                      label={`${result.imageSize}x${result.imageSize}`}
                      stats={result.stats}
                    />
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>

          {/* Generation Results */}
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Generation Latency
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: "block" }}>
              Full generation time for different sampling steps and CFG scales (3 runs each)
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Steps</TableCell>
                    <TableCell align="right">CFG</TableCell>
                    <TableCell align="right">Mean Total</TableCell>
                    <TableCell align="right">P95 Total</TableCell>
                    <TableCell align="right">Mean Step</TableCell>
                    <TableCell align="right">P95 Step</TableCell>
                    <TableCell align="right">Status</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {report.generation.map((result, idx) => {
                    const meetsGoal = result.totalStats.p95 < GOAP_GOAL_THRESHOLD;
                    return (
                      <TableRow key={idx}>
                        <TableCell>{result.steps}</TableCell>
                        <TableCell align="right">{result.cfgScale.toFixed(1)}</TableCell>
                        <TableCell align="right">{formatLatency(result.totalStats.mean)}</TableCell>
                        <TableCell align="right">{formatLatency(result.totalStats.p95)}</TableCell>
                        <TableCell align="right">{formatLatency(result.stats.mean)}</TableCell>
                        <TableCell align="right">{formatLatency(result.stats.p95)}</TableCell>
                        <TableCell align="right">
                          <Chip
                            icon={meetsGoal ? <CheckCircleIcon /> : <ErrorIcon />}
                            label={meetsGoal ? "OK" : "SLOW"}
                            size="small"
                            color={meetsGoal ? "success" : "error"}
                          />
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>

          {/* Summary */}
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Summary
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="text.secondary">
                      Classification Performance
                    </Typography>
                    <Typography variant="h4" sx={{ mt: 1, mb: 1 }}>
                      {formatLatency(report.summary.classificationLatency.mean)}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Mean latency across all image sizes
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="text.secondary">
                      Generation Performance
                    </Typography>
                    <Typography variant="h4" sx={{ mt: 1, mb: 1 }}>
                      {formatLatency(report.summary.generationLatency.mean)}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Mean total time across all configurations
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle1" gutterBottom>
                GOAP Success Metric Comparison
              </Typography>
              <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
                <Chip
                  label={`Goal: <${GOAP_GOAL_THRESHOLD / 1000}s for full generation`}
                  variant="outlined"
                />
                {getGoalStatus()}
              </Box>
              <Box sx={{ position: "relative", pt: 1 }}>
                <LinearProgress
                  variant="determinate"
                  value={Math.min(100, (report.summary.generationLatency.p95 / GOAP_GOAL_THRESHOLD) * 100)}
                  color={report.summary.meetsGoal ? "success" : "error"}
                  sx={{ height: 12, borderRadius: 6 }}
                />
                <Box sx={{ display: "flex", justifyContent: "space-between", mt: 1 }}>
                  <Typography variant="caption" color="text.secondary">0s</Typography>
                  <Typography variant="caption" color="text.secondary" fontWeight="bold">
                    {formatLatency(report.summary.generationLatency.p95)} (P95)
                  </Typography>
                  <Typography variant="caption" color="text.secondary">{GOAP_GOAL_THRESHOLD / 1000}s</Typography>
                </Box>
              </Box>
            </Box>
          </Paper>

          {/* Recommendations */}
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <InfoIcon fontSize="small" color="primary" />
              Performance Recommendations
            </Typography>
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              {getPerformanceRecommendations(report).map((rec, idx) => (
                <Alert key={idx} severity="info" sx={{ py: 1 }}>
                  {rec}
                </Alert>
              ))}
            </Box>
          </Paper>

          {/* Fastest Configurations */}
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Fastest Configurations
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: "block" }}>
              Top 5 fastest generation configurations that meet the GOAP goal
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Rank</TableCell>
                    <TableCell>Steps</TableCell>
                    <TableCell align="right">CFG</TableCell>
                    <TableCell align="right">P95 Total</TableCell>
                    <TableCell align="right">Margin</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {report.generation
                    .filter(r => r.totalStats.p95 < GOAP_GOAL_THRESHOLD)
                    .sort((a, b) => a.totalStats.p95 - b.totalStats.p95)
                    .slice(0, 5)
                    .map((result, idx) => {
                      const margin = GOAP_GOAL_THRESHOLD - result.totalStats.p95;
                      return (
                        <TableRow key={idx}>
                          <TableCell>#{idx + 1}</TableCell>
                          <TableCell>{result.steps}</TableCell>
                          <TableCell align="right">{result.cfgScale.toFixed(1)}</TableCell>
                          <TableCell align="right">{formatLatency(result.totalStats.p95)}</TableCell>
                          <TableCell align="right">
                            <Chip
                              label={`+${formatLatency(margin)}`}
                              size="small"
                              color="success"
                              variant="outlined"
                            />
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  {report.generation.filter(r => r.totalStats.p95 < GOAP_GOAL_THRESHOLD).length === 0 && (
                    <TableRow>
                      <TableCell colSpan={5} align="center">
                        <Typography variant="body2" color="text.secondary">
                          No configurations meet the GOAP goal
                        </Typography>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </div>
      )}

      {!report && !isRunning && (
        <div className="row container" style={{ justifyContent: "center", gap: "2rem" }}>
          <Paper
            elevation={3}
            sx={{
              p: 4,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              textAlign: "center",
              maxWidth: 600
            }}
          >
            <Typography variant="h6" gutterBottom>
              Ready to Benchmark
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Click "Run Benchmark" to measure inference latency for classification and generation.
              The benchmark will test multiple configurations and compare results against GOAP success metrics.
            </Typography>
            <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap", justifyContent: "center" }}>
              <Chip label="Classification: 128x128, 224x224, 256x256" variant="outlined" />
              <Chip label="Generation: 10, 25, 50, 100 steps" variant="outlined" />
              <Chip label="CFG: 1.0, 1.5, 2.0, 3.0" variant="outlined" />
            </Box>
          </Paper>
        </div>
      )}
    </>
  );
}
