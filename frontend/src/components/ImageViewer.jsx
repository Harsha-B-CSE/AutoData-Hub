import React, { useState } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, AlertTriangle, ShieldCheck, Eye, Sparkles } from 'lucide-react';
import { api } from '../services/api';

export default function ImageViewer({ datasetId, previewData }) {
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  if (!previewData || !previewData.images || previewData.images.length === 0) {
    return (
      <div className="p-8 text-center text-slate-500 bg-slate-900/40 rounded-xl border border-slate-800">
        No images available in this visual dataset.
      </div>
    );
  }

  const currentImage = previewData.images[selectedIdx];
  const imageUrl = currentImage
    ? api.getDatasetImageUrl(datasetId, currentImage.relative_path || currentImage.name)
    : '';

  const handleMouseDown = (e) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleResetZoom = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const prof = currentImage?.profile;

  return (
    <div className="space-y-4">
      {/* Top Asset Telemetry Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-900 p-3 rounded-xl border border-slate-800">
        <div className="flex items-center gap-3">
          <span className="text-xs font-semibold text-white px-2.5 py-1 rounded bg-slate-800">
            Image {selectedIdx + 1} of {previewData.total_images}
          </span>
          <span className="text-xs text-slate-300 font-mono truncate max-w-[200px] sm:max-w-xs">
            {currentImage?.name}
          </span>
        </div>

        {/* Zoom Controls */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <span className="text-xs font-mono text-slate-400 w-12 text-center">
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={() => setZoom((z) => Math.min(3, z + 0.25))}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={handleResetZoom}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
            title="Reset Pan/Zoom"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Main Canvas Viewport */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        <div className="lg:col-span-3 bg-slate-950 rounded-xl border border-slate-800 h-[450px] relative overflow-hidden flex items-center justify-center cursor-grab active:cursor-grabbing select-none">
          <div
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            className="w-full h-full flex items-center justify-center"
          >
            {imageUrl && (
              <img
                src={imageUrl}
                alt={currentImage?.name}
                draggable={false}
                style={{
                  transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                  transition: isDragging ? 'none' : 'transform 0.1s ease-out',
                }}
                className="max-h-[90%] max-w-[90%] object-contain rounded shadow-2xl pointer-events-none"
              />
            )}
          </div>

          {/* Quick HUD overlay */}
          {prof && (
            <div className="absolute bottom-3 left-3 bg-slate-900/90 backdrop-blur px-3 py-1.5 rounded-lg border border-slate-800 text-[11px] font-mono text-slate-300 flex items-center gap-3">
              <span>{prof.width}×{prof.height} px</span>
              <span>{prof.color_space}</span>
              <span className={prof.is_blurry ? 'text-amber-400' : 'text-emerald-400'}>
                Var: {prof.laplacian_variance} {prof.is_blurry ? '(Blurry)' : '(Sharp)'}
              </span>
            </div>
          )}
        </div>

        {/* Side Telemetry & Inspection Panel */}
        <div className="space-y-3">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Asset Properties
            </h4>

            {prof ? (
              <div className="space-y-2 text-xs">
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">Dimensions</span>
                  <span className="text-white font-mono">{prof.width} × {prof.height} px</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">Aspect Ratio</span>
                  <span className="text-white font-mono">{prof.aspect_ratio}</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">Color Space</span>
                  <span className="text-white font-mono">{prof.color_space} ({prof.channels} ch)</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">Bit Depth</span>
                  <span className="text-white font-mono">{prof.bit_depth}-bit</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">Laplacian Sharpness</span>
                  <span className={`font-mono font-medium ${prof.is_blurry ? 'text-amber-400' : 'text-emerald-400'}`}>
                    {prof.laplacian_variance}
                  </span>
                </div>

                <div className="pt-2">
                  {prof.is_blurry ? (
                    <div className="p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-400 flex items-start gap-2">
                      <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                      <span>Noticeable blur detected. Sharpness enhancement recommended.</span>
                    </div>
                  ) : (
                    <div className="p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 flex items-start gap-2">
                      <ShieldCheck className="w-4 h-4 flex-shrink-0 mt-0.5" />
                      <span>Sharp focus verified. High-frequency edge preservation confirmed.</span>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-500 italic">Select an asset from the gallery below.</p>
            )}
          </div>
        </div>
      </div>

      {/* Thumbnail Carousel */}
      <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
        <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
          Dataset Asset Gallery ({previewData.images.length} Loaded)
        </h4>
        <div className="flex items-center gap-2 overflow-x-auto pb-2">
          {previewData.images.map((img, idx) => {
            const thumbUrl = api.getDatasetImageUrl(datasetId, img.relative_path || img.name);
            const isSelected = idx === selectedIdx;
            return (
              <div
                key={idx}
                onClick={() => {
                  setSelectedIdx(idx);
                  handleResetZoom();
                }}
                className={`relative flex-shrink-0 w-20 h-20 rounded-lg overflow-hidden border-2 cursor-pointer transition ${
                  isSelected ? 'border-brand-500 scale-105 shadow-md' : 'border-slate-800 hover:border-slate-600 opacity-70 hover:opacity-100'
                }`}
              >
                <img
                  src={thumbUrl}
                  alt={img.name}
                  className="w-full h-full object-cover"
                />
                <span className="absolute bottom-0 inset-x-0 bg-slate-950/80 text-[9px] font-mono text-center truncate text-slate-300 px-1 py-0.5">
                  {img.name}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
