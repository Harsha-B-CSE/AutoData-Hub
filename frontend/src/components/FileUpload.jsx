import React, { useState, useRef } from 'react';
import { UploadCloud, FileSpreadsheet, Image as ImageIcon, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';
import { api } from '../services/api';

export default function FileUpload({ onUploadSuccess }) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragging(true);
    } else if (e.type === 'dragleave') {
      setIsDragging(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = async (file) => {
    setError(null);
    setUploadResult(null);

    // Client-side pre-validation
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    const validExts = ['.csv', '.tsv', '.xls', '.xlsx', '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.zip'];

    if (!validExts.includes(ext)) {
      setError(`Unsupported file format '${ext}'. Please upload CSV, Excel, Image, or ZIP archives.`);
      return;
    }

    if (file.size > 1024 * 1024 * 1024) {
      setError('File size exceeds the 1 GB maximum boundary.');
      return;
    }

    try {
      setUploading(true);
      const res = await api.uploadDataset(file);
      setUploadResult(res);
      if (onUploadSuccess) {
        onUploadSuccess(res);
      }
    } catch (err) {
      setError(err.message || 'Failed to upload dataset.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="w-full">
      <div
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-8 sm:p-12 text-center cursor-pointer transition-all duration-200 ${
          isDragging
            ? 'border-brand-500 bg-brand-500/10 scale-[1.01]'
            : 'border-slate-700 bg-slate-900/60 hover:border-slate-500 hover:bg-slate-900'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          accept=".csv,.tsv,.xls,.xlsx,.jpg,.jpeg,.png,.webp,.bmp,.tiff,.tif,.zip"
          onChange={handleFileChange}
        />

        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="w-16 h-16 rounded-2xl bg-slate-800/80 border border-slate-700 flex items-center justify-center text-brand-500 shadow-inner">
            {uploading ? (
              <Loader2 className="w-8 h-8 animate-spin text-brand-500" />
            ) : (
              <UploadCloud className="w-8 h-8" />
            )}
          </div>

          <div>
            <p className="text-base sm:text-lg font-medium text-slate-200">
              {uploading ? 'Ingesting and Verifying Magic Bytes...' : 'Drag and drop your dataset here, or click to browse'}
            </p>
            <p className="text-sm text-slate-400 mt-1">
              Supports CSV, TSV, Excel (.xlsx/.xls), Images (JPG, PNG, WEBP, BMP, TIFF), or ZIP Archives
            </p>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-2 pt-2">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <FileSpreadsheet className="w-3.5 h-3.5" /> Tabular Data
            </span>
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              <ImageIcon className="w-3.5 h-3.5" /> Vision Assets & ZIPs
            </span>
            <span className="px-3 py-1 rounded-full text-xs font-medium bg-slate-800 text-slate-400 border border-slate-700">
              Max 1 GB • Non-Destructive Isolated
            </span>
          </div>
        </div>
      </div>

      {error && (
        <div className="mt-4 p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 flex items-start gap-3 text-sm">
          <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-medium">Ingestion Error</p>
            <p className="text-red-300/90">{error}</p>
          </div>
        </div>
      )}

      {uploadResult && (
        <div className="mt-4 p-4 rounded-lg bg-brand-500/10 border border-brand-500/20 text-brand-400 flex items-start justify-between text-sm">
          <div className="flex items-start gap-3">
            <CheckCircle2 className="w-5 h-5 flex-shrink-0 mt-0.5 text-brand-500" />
            <div>
              <p className="font-medium text-white">{uploadResult.filename}</p>
              <p className="text-xs text-slate-400">
                Type: <span className="text-brand-400 uppercase font-semibold">{uploadResult.type}</span> • Size: {(uploadResult.size / 1024).toFixed(1)} KB • SHA-256: {uploadResult.sha256?.substring(0, 16)}...
              </p>
            </div>
          </div>
          <span className="text-xs font-mono bg-slate-800 text-slate-300 px-2 py-1 rounded">
            ID: {uploadResult.dataset_id?.substring(0, 8)}
          </span>
        </div>
      )}
    </div>
  );
}
