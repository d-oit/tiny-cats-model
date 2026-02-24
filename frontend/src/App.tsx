import "./App.css";
import Navbar from "./components/Navbar";
import { HashRouter as Router, Routes, Route } from "react-router-dom";
import ClassifyPage from "./pages/classify/ClassifyPage";
import GeneratePage from "./pages/generate/GeneratePage";
import BenchmarkPage from "./pages/benchmark/BenchmarkPage";

import { ThemeProvider } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import { getTheme } from "./theme";
import { useEffect, useMemo, useState } from "react";

function App() {
  const [mode, setMode] = useState<"light" | "dark">(() => {
    try {
      const v = localStorage.getItem("ui-theme");
      return (v === "light" || v === "dark") ? v : "dark";
    } catch {
      return "dark";
    }
  });

  useEffect(() => {
    try { localStorage.setItem("ui-theme", mode); } catch { /* ignore */ }
  }, [mode]);

  const toggleMode = () => setMode(m => (m === "dark" ? "light" : "dark"));

  const theme = useMemo(() => getTheme(mode), [mode]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <div className="navbar">
          <Navbar themeMode={mode} toggleTheme={toggleMode} />
        </div>
        <Routes>
          <Route path="/" element={<ClassifyPage />} />
          <Route path="/classify" element={<ClassifyPage />} />
          <Route path="/generate" element={<GeneratePage />} />
          <Route path="/benchmark" element={<BenchmarkPage />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
