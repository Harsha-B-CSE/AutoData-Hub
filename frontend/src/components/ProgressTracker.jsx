import React, { useEffect, useState } from 'react';
import { Loader2, CheckCircle2, AlertOctagon, Clock, Cpu, HardDrive } from 'lucide-react';
import { api } from '../services/api';

export default function ProgressTracker({ jobId, onJobCompleted }) {
  const [job, setJob] = useState(null);
  const [benchmarks, setBenchmarks] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!jobId) return;

    let intervalId = null;

    const poll = async () => {
      try {
        const data = await api.getJobStatus(jobId);
        setJob(data);

        // Fetch benchmarks
        const bData = await api.getJobBenchmarks(jobId).catch(() => []);
        setBenchmarks(bData);

        if (data.status === 'COMPLETED') {
          clearInterval(intervalId);
          if (onJobCompleted) onJobCompleted(data);
        } else if (data.status === 'FAILED') {
          clearInterval(intervalId);
          setError(data.error_message || 'Processing pipeline failed');
        }
      } catch (err) {
        console.error(err);
      }
    };

    poll();
    intervalId = setInterval(poll, 700);

    return () => clearInterval(intervalId);
  }, [jobId]);

  if (!job) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-center text-slate-400">
        <Loader2 className="w-6 h-6 animate-spin mx-auto text-brand-500 mb-2" />
        <p className="text-xs">Initializing Job Orchestrator...</p>
      </div>
    );
  }

  const isCompleted = job.status === 'COMPLETED';
  const isFailed = job.status === 'FAILED';

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-6">
      {/* Top Status Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2.5">
            {isCompleted ? (
              <CheckCircle2 className="w-5 h-5 text-brand-500" />
            ) : isFailed ? (
              <AlertOctagon className="w-5 h-5 text-red-500" />
            ) : (
              <Loader2 className="w-5 h-5 animate-spin text-brand-500" />
            )}
            <h3 className="text-base font-semibold text-white">
              {isCompleted ? 'Pipeline Completed Successfully' : isFailed ? 'Pipeline Failed' : 'Pipeline Executing...'}
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Job ID: <span className="font-mono text-slate-300">{job.job_id}</span> • Mode: <span className="uppercase text-brand-400 font-semibold">{job.execution_mode}</span>
          </p>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono text-slate-300">
          {job.elapsed_ms !== null && (
            <div className="flex items-center gap-1.5 bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800">
              <Clock className="w-3.5 h-3.5 text-slate-400" />
              <span>{(job.elapsed_ms / 1000).toFixed(2)}s</span>
            </div>
          )}
          <span
            className={`px-3 py-1.5 rounded-lg uppercase font-semibold text-[11px] ${
              isCompleted
                ? 'bg-brand-500/10 text-brand-400 border border-brand-500/20'
                : isFailed
                ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                : 'bg-blue-500/10 text-blue-400 border border-blue-500/20 animate-pulse'
            }`}
          >
            {job.status}
          </span>
        </div>
      </div>

      {/* Progress Bar & Stage Description */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="font-medium text-slate-300 flex items-center gap-2">
            {!isCompleted && !isFailed && <span className="w-2 h-2 rounded-full bg-brand-500 animate-ping" />}
            {job.current_stage || 'Preparing Execution...'}
          </span>
          <span className="font-mono font-semibold text-brand-400">{Math.round(job.progress)}%</span>
        </div>
        <div className="w-full h-2.5 bg-slate-950 rounded-full overflow-hidden border border-slate-800/80 p-0.5">
          <div
            className={`h-full rounded-full transition-all duration-300 ${
              isFailed ? 'bg-red-500' : 'bg-gradient-to-r from-emerald-500 to-brand-500'
            }`}
            style={{ width: `${Math.max(3, job.progress)}%` }}
          />
        </div>
      </div>

      {/* Error Traceback (if failed) */}
      {error && (
        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-xs font-mono space-y-1">
          <p className="font-bold font-sans">Error Traceback:</p>
          <pre className="overflow-x-auto whitespace-pre-wrap">{error}</pre>
        </div>
      )}

      {/* Benchmark Stages Telemetry Feed */}
      {benchmarks.length > 0 && (
        <div className="space-y-3 pt-2">
          <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
            <Cpu className="w-3.5 h-3.5 text-brand-500" /> Stage Latency & Resource Utilization
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5 font-mono text-xs">
            {benchmarks.map((b) => (
              <div key={b.id} className="bg-slate-950 p-2.5 rounded-lg border border-slate-800/80 space-y-1">
                <span className="text-[11px] font-sans font-medium text-slate-300 block truncate">
                  {b.stage_name}
                </span>
                <div className="flex items-center justify-between text-[11px] text-slate-400">
                  <span className="text-brand-400 font-semibold">{b.latency_ms.toFixed(1)} ms</span>
                  <span className="text-slate-500">CPU {b.cpu_percent.toFixed(0)}%</span>
                  <span className={b.memory_delta_mb > 0 ? 'text-amber-400' : 'text-slate-400'}>
                    {b.memory_delta_mb >= 0 ? `+${b.memory_delta_mb.toFixed(1)}` : b.memory_delta_mb.toFixed(1)} MB
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
