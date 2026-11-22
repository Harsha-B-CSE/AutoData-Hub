import React, { useState, useEffect, useRef } from 'react';
import {
  Columns,
  Sparkles,
  Image as ImageIcon,
  Sliders,
  Download,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  SplitSquareVertical,
  Check,
  FileArchive,
  Info,
  CheckCircle2,
  XCircle,
  Activity,
  Layers,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { api } from '../services/api';

export default function DiffViewer({ jobId }) {
  const [diffData, setDiffData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [sliderPos, setSliderPos] = useState(50); // percentage 0-100
  const [selectedImgIdx, setSelectedImgIdx] = useState(0);
  const [viewMode, setViewMode] = useState('slider'); // 'slider' | 'side-by-side'

  // Dedicated interaction separation: sliding vs panning
  const [isSliding, setIsSliding] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });

  // Zoom & Pan state (synchronized across comparisons)
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });

  // Download state
  const [exportFormat, setExportFormat] = useState('png'); // 'png' | 'jpg'
  const [downloadSuccess, setDownloadSuccess] = useState(false);
  const [showIntelligence, setShowIntelligence] = useState(true);

  const containerRef = useRef(null);

  useEffect(() => {
    if (!jobId) return;
    loadDiff();
  }, [jobId]);

  const loadDiff = async () => {
    try {
      setLoading(true);
      const data = await api.getJobDiff(jobId, 50);
      setDiffData(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // =========================================================================
  // Divider Slider Drag Handlers (Controls ONLY the comparison boundary)
  // =========================================================================
  const handleDividerPointerDown = (e) => {
    e.stopPropagation();
    e.preventDefault();
    setIsSliding(true);
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const handleDividerPointerMove = (e) => {
    if (!isSliding || !containerRef.current) return;
    e.stopPropagation();
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPos(Number(percentage.toFixed(1)));
  };

  const handleDividerPointerUp = (e) => {
    if (!isSliding) return;
    e.stopPropagation();
    setIsSliding(false);
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch (_) {}
  };

  // =========================================================================
  // Canvas Pan Handlers (Controls ONLY image translation / pan)
  // =========================================================================
  const handleCanvasPointerDown = (e) => {
    // Only left click, and do not trigger if dragging the divider
    if (e.button !== 0 || isSliding) return;
    setIsPanning(true);
    setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const handleCanvasPointerMove = (e) => {
    if (!isPanning || isSliding) return;
    setPan({
      x: e.clientX - panStart.x,
      y: e.clientY - panStart.y,
    });
  };

  const handleCanvasPointerUp = (e) => {
    if (!isPanning) return;
    setIsPanning(false);
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch (_) {}
  };

  const handleWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.15 : -0.15;
    setZoom((z) => Math.min(4, Math.max(0.5, Number((z + delta).toFixed(2)))));
  };

  const handleResetZoom = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const handleDownload = () => {
    if (!diffData || !diffData.comparisons || !diffData.comparisons[selectedImgIdx]) return;
    const activePair = diffData.comparisons[selectedImgIdx];
    const url = api.getImageDownloadUrl(jobId, exportFormat, activePair.processed_name);

    const link = document.createElement('a');
    link.href = url;
    link.download = `${activePair.original_name.replace(/\.[^/.]+$/, '')}_processed.${exportFormat}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    setDownloadSuccess(true);
    setTimeout(() => setDownloadSuccess(false), 2500);
  };

  if (loading) {
    return (
      <div className="bg-slate-900/60 backdrop-blur-md border border-slate-800 rounded-2xl p-12 text-center text-slate-400">
        <div className="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
        <p className="text-xs font-mono text-slate-300">Loading comparative before/after verification diffs...</p>
      </div>
    );
  }

  if (!diffData) {
    return (
      <div className="bg-slate-900/60 backdrop-blur-md border border-slate-800 rounded-2xl p-12 text-center text-slate-500">
        No comparative diff data available for this job run.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* ========================================================================= */}
      {/* 1. Tabular Diff Mode */}
      {/* ========================================================================= */}
      {diffData.type === 'tabular' && (
        <div className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-slate-900/80 backdrop-blur-md p-4 rounded-2xl border border-slate-800/80 shadow-lg">
            <div>
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Columns className="w-4 h-4 text-brand-400" /> Tabular Data Hygiene & Transformation Diff
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Comparing raw input vs cleaned dataset. Missing values imputed and modified cells are highlighted.
              </p>
            </div>
            <div className="flex items-center gap-3 text-xs font-mono">
              <span className="bg-slate-950/80 px-2.5 py-1 rounded-lg border border-slate-800 text-slate-400">
                Rows: <strong className="text-white">{diffData.total_before}</strong> → <strong className="text-brand-400">{diffData.total_after}</strong>
              </span>
              <span className="bg-slate-950/80 px-2.5 py-1 rounded-lg border border-slate-800 text-slate-400">
                Cols: <strong className="text-white">{diffData.columns_before?.length}</strong> → <strong className="text-brand-400">{diffData.columns_after?.length}</strong>
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Raw Input Table */}
            <div className="space-y-2">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
                Raw Input Sample (Pre-Processing)
              </span>
              <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/70 max-h-[420px]">
                <table className="w-full text-left border-collapse text-xs">
                  <thead className="bg-slate-900 sticky top-0 border-b border-slate-800">
                    <tr>
                      <th className="p-2.5 text-slate-400 font-medium">#</th>
                      {diffData.columns_before?.map((c) => (
                        <th key={c} className="p-2.5 text-slate-300 font-medium truncate max-w-[120px]">
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/50 font-mono">
                    {diffData.rows_before?.map((row, idx) => (
                      <tr key={idx} className="hover:bg-slate-900/40">
                        <td className="p-2 text-slate-500">{idx + 1}</td>
                        {diffData.columns_before?.map((c) => {
                          const val = row[c];
                          const isNull = val === null || val === undefined;
                          return (
                            <td
                              key={c}
                              className={`p-2 truncate max-w-[120px] ${
                                isNull ? 'bg-amber-500/10 text-amber-400 italic' : 'text-slate-300'
                              }`}
                            >
                              {isNull ? 'null' : String(val)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Cleaned Output Table */}
            <div className="space-y-2">
              <span className="text-xs font-semibold text-brand-400 uppercase tracking-wider block flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5" /> Cleaned & Standardized (Post-Processing)
              </span>
              <div className="overflow-x-auto rounded-xl border border-brand-500/20 bg-slate-950/70 max-h-[420px]">
                <table className="w-full text-left border-collapse text-xs">
                  <thead className="bg-slate-900 sticky top-0 border-b border-brand-500/20">
                    <tr>
                      <th className="p-2.5 text-slate-400 font-medium">#</th>
                      {diffData.columns_after?.map((c) => (
                        <th key={c} className="p-2.5 text-brand-300 font-medium truncate max-w-[120px]">
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/50 font-mono">
                    {diffData.rows_after?.map((row, idx) => (
                      <tr key={idx} className="hover:bg-slate-900/40">
                        <td className="p-2 text-slate-500">{idx + 1}</td>
                        {diffData.columns_after?.map((c) => {
                          const val = row[c];
                          const isNull = val === null || val === undefined;
                          const wasNull = diffData.rows_before?.[idx]?.[c] === null || diffData.rows_before?.[idx]?.[c] === undefined;
                          return (
                            <td
                              key={c}
                              className={`p-2 truncate max-w-[120px] ${
                                wasNull && !isNull
                                  ? 'bg-emerald-500/15 text-emerald-300 font-semibold'
                                  : 'text-slate-300'
                              }`}
                            >
                              {isNull ? 'null' : (typeof val === 'number' ? Number(val.toFixed(4)) : String(val))}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* 2. Vision Diff Mode: High-Precision Split Slider & Side-by-Side with Zoom */}
      {/* ========================================================================= */}
      {diffData.type === 'vision' && diffData.comparisons?.length > 0 && (
        <div className="space-y-4">
          {/* Main Inspection Control Bar */}
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 bg-slate-900/80 backdrop-blur-md p-4 rounded-2xl border border-slate-800/80 shadow-lg">
            <div className="space-y-1">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <ImageIcon className="w-4 h-4 text-brand-400" /> Interactive Before / After Image Verification
              </h3>
              <p className="text-xs text-slate-400">
                Inspect adaptive luminance recovery, smart LAB contrast, edge-preserving denoising, and authentic details.
              </p>
            </div>

            {/* View Mode & Zoom Controls */}
            <div className="flex flex-wrap items-center gap-2">
              {/* View Mode Toggle */}
              <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800">
                <button
                  onClick={() => setViewMode('slider')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                    viewMode === 'slider'
                      ? 'bg-brand-500 text-white shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                  title="Split Slider View"
                >
                  <Sliders className="w-3.5 h-3.5" />
                  <span>Split Slider</span>
                </button>
                <button
                  onClick={() => setViewMode('side-by-side')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                    viewMode === 'side-by-side'
                      ? 'bg-brand-500 text-white shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                  title="Side-by-Side Synchronized View"
                >
                  <SplitSquareVertical className="w-3.5 h-3.5" />
                  <span>Side-by-Side</span>
                </button>
              </div>

              {/* Synchronized Zoom Controls */}
              <div className="flex items-center bg-slate-950 px-2 py-1 rounded-xl border border-slate-800 gap-1">
                <button
                  onClick={() => setZoom((z) => Math.max(0.5, Number((z - 0.25).toFixed(2))))}
                  className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
                  title="Zoom Out (Scroll Down)"
                >
                  <ZoomOut className="w-3.5 h-3.5" />
                </button>
                <span className="text-xs font-mono text-slate-300 w-12 text-center select-none">
                  {Math.round(zoom * 100)}%
                </span>
                <button
                  onClick={() => setZoom((z) => Math.min(4, Number((z + 0.25).toFixed(2))))}
                  className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
                  title="Zoom In (Scroll Up)"
                >
                  <ZoomIn className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={handleResetZoom}
                  className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
                  title="Reset Zoom & Pan"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </div>

          {/* Asset Switcher (Thumbnails / Badges) */}
          {diffData.comparisons.length > 1 && (
            <div className="flex items-center gap-2 overflow-x-auto pb-1.5 pt-0.5">
              <span className="text-xs font-medium text-slate-400 shrink-0">Select Asset:</span>
              {diffData.comparisons.map((c, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setSelectedImgIdx(idx);
                    handleResetZoom();
                  }}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-mono transition shrink-0 border ${
                    idx === selectedImgIdx
                      ? 'bg-brand-500/10 border-brand-500/40 text-brand-300 font-semibold shadow-sm'
                      : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700'
                  }`}
                >
                  <span className="w-2 h-2 rounded-full bg-brand-400" />
                  <span className="truncate max-w-[140px]">{c.original_name}</span>
                </button>
              ))}
            </div>
          )}

          {/* Main Inspection Canvas Viewport */}
          {(() => {
            const activePair = diffData.comparisons[selectedImgIdx];
            if (!activePair) return null;
            const enh = activePair.enhancements || {};

            return (
              <div className="space-y-4">
                {/* Viewport Canvas with Clean Separation of Slider Drag and Canvas Pan */}
                <div
                  ref={containerRef}
                  onWheel={handleWheel}
                  onPointerDown={handleCanvasPointerDown}
                  onPointerMove={handleCanvasPointerMove}
                  onPointerUp={handleCanvasPointerUp}
                  onPointerCancel={handleCanvasPointerUp}
                  className={`relative w-full h-[520px] bg-slate-950 rounded-2xl border border-slate-800/80 overflow-hidden select-none shadow-2xl touch-none ${
                    isPanning ? 'cursor-grabbing' : 'cursor-grab'
                  }`}
                >
                  {/* Mode A: Interactive Split Slider with Pixel-Perfect Clip-Path Alignment */}
                  {viewMode === 'slider' && (
                    <div className="w-full h-full relative overflow-hidden pointer-events-none">
                      {/* Processed (After) Image - Base Layer */}
                      <div
                        className="absolute inset-0 flex items-center justify-center pointer-events-none select-none"
                        style={{
                          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                          transition: isPanning ? 'none' : 'transform 0.08s ease-out',
                        }}
                      >
                        <img
                          src={activePair.after_url}
                          alt="Enhanced Output"
                          className="max-w-full max-h-full object-contain pointer-events-none drop-shadow-2xl"
                          draggable={false}
                        />
                      </div>

                      {/* Original (Before) Image - Clipped strictly with CSS clip-path */}
                      <div
                        className="absolute inset-0 flex items-center justify-center pointer-events-none select-none"
                        style={{
                          clipPath: `inset(0 calc(100% - ${sliderPos}%) 0 0)`,
                          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                          transition: isPanning ? 'none' : 'transform 0.08s ease-out',
                        }}
                      >
                        <img
                          src={activePair.before_url}
                          alt="Original Input"
                          className="max-w-full max-h-full object-contain pointer-events-none drop-shadow-2xl"
                          draggable={false}
                        />
                      </div>

                      {/* Divider Handle: Isolated pointer events, moves ONLY comparison boundary */}
                      <div
                        onPointerDown={handleDividerPointerDown}
                        onPointerMove={handleDividerPointerMove}
                        onPointerUp={handleDividerPointerUp}
                        onPointerCancel={handleDividerPointerUp}
                        style={{ left: `${sliderPos}%` }}
                        className="absolute top-0 bottom-0 -ml-5 w-10 z-30 cursor-ew-resize flex items-center justify-center touch-none select-none pointer-events-auto group"
                        title="Drag left/right to compare before and after"
                      >
                        {/* Vertical glowing divider line */}
                        <div className="w-0.5 h-full bg-white shadow-[0_0_12px_rgba(255,255,255,0.9)] group-hover:w-1 group-active:w-1 transition-all" />

                        {/* Thumb grip button */}
                        <div className="absolute top-1/2 -translate-y-1/2 w-9 h-9 rounded-full bg-slate-900 border-2 border-white flex items-center justify-center shadow-2xl group-hover:scale-110 active:scale-95 transition-transform">
                          <Sliders className="w-4 h-4 text-white" />
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Mode B: Side-by-Side Synchronized View */}
                  {viewMode === 'side-by-side' && (
                    <div className="w-full h-full grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-slate-800">
                      {/* Left: Original */}
                      <div className="relative w-full h-full flex items-center justify-center overflow-hidden bg-slate-950/80 pointer-events-none">
                        <div
                          className="absolute inset-0 flex items-center justify-center pointer-events-none select-none"
                          style={{
                            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                            transition: isPanning ? 'none' : 'transform 0.08s ease-out',
                          }}
                        >
                          <img
                            src={activePair.before_url}
                            alt="Original Input"
                            className="max-w-[90%] max-h-[90%] object-contain pointer-events-none shadow-xl"
                            draggable={false}
                          />
                        </div>
                        <span className="absolute top-3 left-3 bg-slate-900/90 backdrop-blur text-slate-300 text-[10px] font-mono uppercase px-2.5 py-1 rounded-md border border-slate-700 pointer-events-none shadow-md">
                          Original Input
                        </span>
                      </div>

                      {/* Right: Enhanced */}
                      <div className="relative w-full h-full flex items-center justify-center overflow-hidden bg-slate-950/80 pointer-events-none">
                        <div
                          className="absolute inset-0 flex items-center justify-center pointer-events-none select-none"
                          style={{
                            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                            transition: isPanning ? 'none' : 'transform 0.08s ease-out',
                          }}
                        >
                          <img
                            src={activePair.after_url}
                            alt="Enhanced Output"
                            className="max-w-[90%] max-h-[90%] object-contain pointer-events-none shadow-xl"
                            draggable={false}
                          />
                        </div>
                        <span className="absolute top-3 right-3 bg-brand-500/90 backdrop-blur text-white text-[10px] font-mono uppercase px-2.5 py-1 rounded-md border border-brand-400 pointer-events-none shadow-md">
                          Enhanced Output
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Mode A Overlay Badges */}
                  {viewMode === 'slider' && (
                    <>
                      <div className="absolute top-3 left-3 bg-slate-900/90 backdrop-blur text-slate-300 text-[10px] font-mono uppercase px-2.5 py-1 rounded-md border border-slate-700 pointer-events-none shadow-md">
                        Original Input (Left)
                      </div>
                      <div className="absolute top-3 right-3 bg-brand-500/90 backdrop-blur text-white text-[10px] font-mono uppercase px-2.5 py-1 rounded-md border border-brand-400 pointer-events-none shadow-md">
                        Enhanced Output (Right)
                      </div>
                    </>
                  )}

                  {/* Interaction Instructions Overlay Pill */}
                  <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-slate-900/85 backdrop-blur-md px-3.5 py-1.5 rounded-full border border-slate-800 text-[10px] font-mono text-slate-400 pointer-events-none shadow-lg whitespace-nowrap">
                    Scroll wheel zooms • Drag canvas pans • Drag divider compares
                  </div>
                </div>

                {/* ================================================================= */}
                {/* Adaptive Enhancement Intelligence Panel (Operations Applied & Why) */}
                {/* ================================================================= */}
                {enh && (enh.operations_applied || enh.skipped_operations) && (
                  <div className="bg-slate-900/80 backdrop-blur-md rounded-2xl border border-slate-800/80 overflow-hidden shadow-lg">
                    <button
                      type="button"
                      onClick={() => setShowIntelligence(!showIntelligence)}
                      className="w-full p-4 flex items-center justify-between text-left hover:bg-slate-850/50 transition"
                    >
                      <div className="flex items-center gap-2.5">
                        <Activity className="w-4 h-4 text-brand-400" />
                        <div>
                          <h4 className="text-xs font-semibold text-white tracking-wide flex items-center gap-2">
                            <span>Adaptive Enhancement Intelligence</span>
                            {enh.preset_name && (
                              <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-full bg-brand-500/10 text-brand-300 border border-brand-500/30">
                                {enh.preset_name} Preset
                              </span>
                            )}
                          </h4>
                          <p className="text-[11px] text-slate-400 mt-0.5">{enh.assessment}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 text-slate-400 text-xs font-mono">
                        <span>{enh.operations_applied?.length || 0} Applied</span>
                        <span>•</span>
                        <span>{enh.skipped_operations?.length || 0} Skipped</span>
                        {showIntelligence ? <ChevronUp className="w-4 h-4 ml-1" /> : <ChevronDown className="w-4 h-4 ml-1" />}
                      </div>
                    </button>

                    {showIntelligence && (
                      <div className="p-4 pt-0 space-y-4 border-t border-slate-800/60 text-xs">
                        {/* Telemetry Comparison Table */}
                        {enh.metrics_before && enh.metrics_after && (
                          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-3">
                            <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                              <span className="text-[10px] text-slate-400 block font-mono">Luminance (Brightness)</span>
                              <div className="text-xs font-mono mt-1">
                                <span className="text-slate-400">{enh.metrics_before.mean_luminance}</span>
                                <span className="text-slate-600 mx-1.5">→</span>
                                <span className="text-brand-400 font-semibold">{enh.metrics_after.mean_luminance}</span>
                              </div>
                            </div>
                            <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                              <span className="text-[10px] text-slate-400 block font-mono">Contrast Std (Range)</span>
                              <div className="text-xs font-mono mt-1">
                                <span className="text-slate-400">{enh.metrics_before.std_luminance}</span>
                                <span className="text-slate-600 mx-1.5">→</span>
                                <span className="text-brand-400 font-semibold">{enh.metrics_after.std_luminance}</span>
                              </div>
                            </div>
                            <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                              <span className="text-[10px] text-slate-400 block font-mono">Noise Sigma</span>
                              <div className="text-xs font-mono mt-1">
                                <span className="text-slate-400">{enh.metrics_before.noise_sigma}</span>
                                <span className="text-slate-600 mx-1.5">→</span>
                                <span className="text-emerald-400 font-semibold">{enh.metrics_after.noise_sigma}</span>
                              </div>
                            </div>
                            <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                              <span className="text-[10px] text-slate-400 block font-mono">Laplacian Sharpness</span>
                              <div className="text-xs font-mono mt-1">
                                <span className="text-slate-400">{enh.metrics_before.sharpness}</span>
                                <span className="text-slate-600 mx-1.5">→</span>
                                <span className="text-brand-400 font-semibold">{enh.metrics_after.sharpness}</span>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Applied Operations */}
                        {enh.operations_applied?.length > 0 && (
                          <div className="space-y-2">
                            <span className="text-[10px] uppercase font-semibold text-emerald-400 tracking-wider flex items-center gap-1.5">
                              <CheckCircle2 className="w-3.5 h-3.5" /> Operations Executed
                            </span>
                            <div className="space-y-1.5">
                              {enh.operations_applied.map((op, i) => (
                                <div key={i} className="p-2.5 bg-slate-950/70 rounded-xl border border-emerald-500/20 flex flex-col sm:flex-row sm:items-center justify-between gap-1 text-xs">
                                  <div>
                                    <span className="font-semibold text-slate-200">{op.name || op.operation}</span>
                                    <p className="text-slate-400 text-[11px] mt-0.5">{op.reason}</p>
                                  </div>
                                  {op.parameters && (
                                    <span className="font-mono text-[10px] text-brand-300 bg-brand-500/10 px-2 py-0.5 rounded shrink-0">
                                      {Object.entries(op.parameters).map(([k, v]) => `${k}=${v}`).join(', ')}
                                    </span>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Skipped Operations */}
                        {enh.skipped_operations?.length > 0 && (
                          <div className="space-y-2">
                            <span className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider flex items-center gap-1.5">
                              <Info className="w-3.5 h-3.5 text-slate-500" /> Preserved Intact (Operations Skipped)
                            </span>
                            <div className="space-y-1.5">
                              {enh.skipped_operations.map((op, i) => (
                                <div key={i} className="p-2 bg-slate-950/40 rounded-xl border border-slate-800/60 flex items-start gap-2 text-xs text-slate-400">
                                  <span className="text-slate-500 font-mono text-[10px] shrink-0 mt-0.5">SKIP:</span>
                                  <span>{op.reason}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* ================================================================= */}
                {/* Image-Specific Download Bar (Visual Datasets Only) */}
                {/* ================================================================= */}
                <div className="bg-slate-900/80 backdrop-blur-md p-4 rounded-2xl border border-slate-800/80 shadow-lg flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  <div className="space-y-0.5">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold text-white">Download Processed Image:</span>
                      <span className="text-xs font-mono text-brand-400 font-medium truncate max-w-[200px] sm:max-w-xs">
                        {activePair.original_name}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400">
                      Raster output identically matches the displayed preview.
                    </p>
                  </div>

                  <div className="flex flex-wrap items-center gap-2.5">
                    {/* Format Selector */}
                    <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800">
                      <button
                        onClick={() => setExportFormat('png')}
                        className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition ${
                          exportFormat === 'png'
                            ? 'bg-brand-500 text-white shadow-sm'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        PNG (Lossless)
                      </button>
                      <button
                        onClick={() => setExportFormat('jpg')}
                        className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition ${
                          exportFormat === 'jpg'
                            ? 'bg-brand-500 text-white shadow-sm'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        JPEG (.jpg)
                      </button>
                    </div>

                    {/* Primary Download Button */}
                    <button
                      onClick={handleDownload}
                      className="flex items-center gap-2 px-5 py-2 rounded-xl bg-brand-500 hover:bg-brand-600 active:scale-95 text-white font-medium text-xs transition shadow-lg shadow-brand-500/25"
                    >
                      {downloadSuccess ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-white animate-bounce" />
                          <span>Downloaded!</span>
                        </>
                      ) : (
                        <>
                          <Download className="w-3.5 h-3.5 text-white" />
                          <span>Download Image ({exportFormat.toUpperCase()})</span>
                        </>
                      )}
                    </button>

                    {/* Export Full ZIP bundle option */}
                    <a
                      href={api.getJobExportUrl(jobId, 'zip')}
                      className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white text-xs font-medium transition border border-slate-700/60"
                      title="Download complete standardized dataset archive"
                    >
                      <FileArchive className="w-3.5 h-3.5 text-slate-400" />
                      <span>All (ZIP)</span>
                    </a>
                  </div>
                </div>
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
}
