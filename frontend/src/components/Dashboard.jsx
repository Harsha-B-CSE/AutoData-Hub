import React, { useState, useEffect } from 'react';
import {
  Layers,
  UploadCloud,
  Eye,
  Sliders,
  Activity,
  History,
  CheckCircle2,
  FileSpreadsheet,
  Image as ImageIcon,
  Cpu,
  RefreshCw,
  HardDrive
} from 'lucide-react';
import { api } from '../services/api';
import FileUpload from './FileUpload';
import TabularGrid from './TabularGrid';
import ImageViewer from './ImageViewer';
import ConfigPanel from './ConfigPanel';
import ProgressTracker from './ProgressTracker';
import DiffViewer from './DiffViewer';
import QualitySummary from './QualitySummary';
import ThreeHeroCanvas from './ThreeHeroCanvas';

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' | 'preview' | 'config' | 'tracker' | 'diff' | 'quality' | 'history'
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [previewData, setPreviewData] = useState(null);
  const [previewPage, setPreviewPage] = useState(1);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [currentJobId, setCurrentJobId] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      const [h, dsList, jList] = await Promise.all([
        api.getHealth().catch(() => null),
        api.listDatasets().catch(() => []),
        api.listJobs().catch(() => []),
      ]);
      setHealth(h);
      setDatasets(dsList);
      setJobs(jList);

      if (dsList.length > 0 && !selectedDataset) {
        selectDataset(dsList[0]);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const selectDataset = async (ds, targetPage = 1) => {
    setSelectedDataset(ds);
    setPreviewLoading(true);
    try {
      const pData = await api.previewDataset(ds.id, targetPage, 50);
      setPreviewData(pData);
      setPreviewPage(targetPage);
    } catch (err) {
      console.error(err);
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleUploadSuccess = async (uploadRes) => {
    await loadInitialData();
    const newDs = {
      id: uploadRes.dataset_id,
      filename: uploadRes.filename,
      file_type: uploadRes.type,
      file_size_bytes: uploadRes.size,
      sha256: uploadRes.sha256,
    };
    setSelectedDataset(newDs);
    await selectDataset(newDs, 1);
    setActiveTab('preview');
  };

  const handleJobDispatched = (jobId) => {
    setCurrentJobId(jobId);
    setActiveTab('tracker');
    api.listJobs().then(setJobs).catch(() => {});
  };

  const handleJobCompleted = (job) => {
    api.listJobs().then(setJobs).catch(() => {});
  };

  const handleSelectJobFromHistory = (job) => {
    setCurrentJobId(job.job_id);
    if (job.status === 'COMPLETED') {
      setActiveTab('quality');
    } else {
      setActiveTab('tracker');
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <header className="relative border-b border-slate-800/80 bg-slate-900/80 backdrop-blur sticky top-0 z-30 px-4 sm:px-8 py-3.5 flex items-center justify-between overflow-hidden">
        <ThreeHeroCanvas />
        <div className="relative z-10 flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-brand-600 to-emerald-400 flex items-center justify-center text-white shadow-lg shadow-brand-500/20 font-bold text-lg">
            A
          </div>
          <div>
            <h1 className="text-base sm:text-lg font-bold tracking-tight text-white flex items-center gap-2">
              AUTODATA-HUB
              <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-brand-500/10 text-brand-400 border border-brand-500/20">
                Core v1.0
              </span>
            </h1>
            <p className="text-[11px] text-slate-400">
              Dual-Engine Dataset Preprocessing & Performance Optimization Platform
            </p>
          </div>
        </div>

        {/* System Health / Hardware Status */}
        <div className="relative z-10 hidden md:flex items-center gap-4 text-xs">
          {health && (
            <div className="flex items-center gap-2 bg-slate-950/80 backdrop-blur px-3 py-1.5 rounded-lg border border-slate-800 text-slate-300">
              <Cpu className="w-3.5 h-3.5 text-brand-500" />
              <span>
                Workers: <strong className="text-white">{health.worker_cores_allocated}</strong>
              </span>
              <span className="text-slate-600">•</span>
              <span>Max Upload: {health.max_upload_mb} MB</span>
            </div>
          )}
          <div className="flex items-center gap-1.5 text-emerald-400 font-medium">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span>Ready</span>
          </div>
        </div>
      </header>

      {/* Main Layout Container */}
      <div className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-6">
        {/* Dataset Quick Switcher / Status Bar */}
        {selectedDataset && (
          <div className="bg-slate-900/60 p-3.5 rounded-xl border border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2.5">
              {selectedDataset.file_type === 'tabular' ? (
                <FileSpreadsheet className="w-4 h-4 text-emerald-400" />
              ) : (
                <ImageIcon className="w-4 h-4 text-indigo-400" />
              )}
              <span className="text-slate-400 font-medium">Active Asset:</span>
              <span className="text-white font-semibold font-mono">{selectedDataset.filename}</span>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-slate-800 text-slate-300">
                {selectedDataset.file_type}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-slate-500 font-mono">
                {selectedDataset.sha256 ? `${selectedDataset.sha256.substring(0, 16)}...` : ''}
              </span>
              <button
                onClick={() => setActiveTab('upload')}
                className="text-xs text-brand-400 hover:text-brand-300 font-medium px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 transition"
              >
                Switch Asset
              </button>
            </div>
          </div>
        )}

        {/* Navigation Tabs */}
        <div className="flex items-center gap-1.5 overflow-x-auto border-b border-slate-800 pb-2">
          {[
            { id: 'upload', label: 'Ingestion & Upload', icon: UploadCloud },
            { id: 'preview', label: 'Dataset Inspection', icon: Eye, disabled: !selectedDataset },
            { id: 'config', label: 'Pipeline Config', icon: Sliders, disabled: !selectedDataset },
            { id: 'tracker', label: 'Progress & Telemetry', icon: Activity, disabled: !currentJobId },
            { id: 'diff', label: 'Before/After Diff', icon: Layers, disabled: !currentJobId },
            { id: 'quality', label: 'Quality Score & Export', icon: CheckCircle2, disabled: !currentJobId },
            { id: 'history', label: `Execution History (${jobs.length})`, icon: History },
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                disabled={tab.disabled}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-medium whitespace-nowrap transition ${
                  isActive
                    ? 'bg-brand-500 text-white shadow-md shadow-brand-500/20'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900 disabled:opacity-30 disabled:hover:bg-transparent disabled:cursor-not-allowed'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Tab 1: Ingestion & Upload */}
        {activeTab === 'upload' && (
          <div className="space-y-6">
            <FileUpload onUploadSuccess={handleUploadSuccess} />

            {/* Existing Datasets List */}
            {datasets.length > 0 && (
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-white uppercase tracking-wider">
                    Previously Ingested Isolated Datasets ({datasets.length})
                  </h3>
                  <button
                    onClick={loadInitialData}
                    className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs"
                    title="Refresh List"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                  </button>
                </div>

                <div className="divide-y divide-slate-800/80 font-mono text-xs">
                  {datasets.map((ds) => (
                    <div
                      key={ds.id}
                      className="py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2 hover:bg-slate-850/50 px-2 rounded-lg transition"
                    >
                      <div className="flex items-center gap-3">
                        {ds.file_type === 'tabular' ? (
                          <FileSpreadsheet className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                        ) : (
                          <ImageIcon className="w-4 h-4 text-indigo-400 flex-shrink-0" />
                        )}
                        <div>
                          <p className="font-semibold text-slate-200 font-sans">{ds.filename}</p>
                          <p className="text-[11px] text-slate-500">
                            Size: {(ds.file_size_bytes / 1024).toFixed(1)} KB • SHA: {ds.sha256?.substring(0, 16)}...
                          </p>
                        </div>
                      </div>

                      <button
                        onClick={() => {
                          selectDataset(ds, 1);
                          setActiveTab('preview');
                        }}
                        className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-brand-500 hover:text-white text-slate-300 font-sans transition self-start sm:self-auto"
                      >
                        Inspect & Configure →
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Dataset Preview & Inspection */}
        {activeTab === 'preview' && selectedDataset && (
          <div className="space-y-4">
            {selectedDataset.file_type === 'tabular' ? (
              <TabularGrid
                previewData={previewData}
                loading={previewLoading}
                onPageChange={(p) => selectDataset(selectedDataset, p)}
              />
            ) : (
              <ImageViewer datasetId={selectedDataset.id} previewData={previewData} />
            )}
          </div>
        )}

        {/* Tab 3: Configuration Panel */}
        {activeTab === 'config' && selectedDataset && (
          <ConfigPanel
            engineType={selectedDataset.file_type}
            datasetId={selectedDataset.id}
            previewData={previewData}
            onJobDispatched={handleJobDispatched}
          />
        )}

        {/* Tab 4: Progress Tracker */}
        {activeTab === 'tracker' && currentJobId && (
          <ProgressTracker jobId={currentJobId} onJobCompleted={handleJobCompleted} />
        )}

        {/* Tab 5: Before / After Diff */}
        {activeTab === 'diff' && currentJobId && (
          <DiffViewer jobId={currentJobId} />
        )}

        {/* Tab 6: Quality Report & Exports */}
        {activeTab === 'quality' && currentJobId && (
          <QualitySummary jobId={currentJobId} datasetType={selectedDataset?.file_type} />
        )}

        {/* Tab 7: Historical Jobs Catalog */}
        {activeTab === 'history' && (
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white uppercase tracking-wider">
                Historical Execution Audit Log ({jobs.length})
              </h3>
              <button
                onClick={loadInitialData}
                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs"
                title="Refresh List"
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
            </div>

            <div className="overflow-x-auto rounded-xl border border-slate-800">
              <table className="w-full text-left border-collapse text-xs">
                <thead className="bg-slate-950 border-b border-slate-800 text-slate-400">
                  <tr>
                    <th className="p-3">Job ID</th>
                    <th className="p-3">Dataset</th>
                    <th className="p-3">Engine</th>
                    <th className="p-3">Mode</th>
                    <th className="p-3">Status</th>
                    <th className="p-3">Created</th>
                    <th className="p-3 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 font-mono">
                  {jobs.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="p-6 text-center text-slate-500 font-sans">
                        No processing jobs executed yet.
                      </td>
                    </tr>
                  ) : (
                    jobs.map((j) => (
                      <tr key={j.job_id} className="hover:bg-slate-850/50 transition">
                        <td className="p-3 text-slate-400">{j.job_id.substring(0, 8)}...</td>
                        <td className="p-3 font-sans text-slate-200 font-medium">{j.filename}</td>
                        <td className="p-3 uppercase text-slate-400">{j.file_type}</td>
                        <td className="p-3 uppercase text-brand-400">{j.execution_mode}</td>
                        <td className="p-3">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                              j.status === 'COMPLETED'
                                ? 'bg-brand-500/10 text-brand-400'
                                : j.status === 'FAILED'
                                ? 'bg-red-500/10 text-red-400'
                                : 'bg-blue-500/10 text-blue-400'
                            }`}
                          >
                            {j.status}
                          </span>
                        </td>
                        <td className="p-3 text-slate-500">{new Date(j.created_at).toLocaleTimeString()}</td>
                        <td className="p-3 text-right">
                          <button
                            onClick={() => handleSelectJobFromHistory(j)}
                            className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 font-sans text-xs transition"
                          >
                            View Telemetry →
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
