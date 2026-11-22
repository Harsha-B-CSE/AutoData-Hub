import React, { useState, useEffect } from 'react';
import {
  Award,
  CheckCircle2,
  Download,
  FileText,
  FileSpreadsheet,
  Archive,
  Code,
  TrendingUp,
  AlertCircle,
  HelpCircle
} from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend } from 'recharts';
import { api } from '../services/api';

export default function QualitySummary({ jobId, datasetType = null }) {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [recommendations, setRecommendations] = useState([]);

  useEffect(() => {
    if (!jobId) return;
    loadQuality();
  }, [jobId]);

  const loadQuality = async () => {
    try {
      setLoading(true);
      const data = await api.getJobQuality(jobId);
      setReport(data);
      setRecommendations(data.recommendations || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const toggleRecommendation = (id) => {
    setRecommendations((prev) =>
      prev.map((r) => (r.id === id ? { ...r, approved: !r.approved } : r))
    );
  };

  if (loading) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 text-center text-slate-400">
        <p className="text-xs">Computing Quality Analytics and Verification Matrices...</p>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 text-center text-slate-500">
        No quality report found for this execution.
      </div>
    );
  }

  const { pre_score, post_score, score_delta, metrics } = report;
  const isVision = datasetType === 'vision' || report.dataset_type === 'vision' || metrics?.engine_type === 'vision';
  const preQ = metrics?.pre_quality || {};
  const postQ = metrics?.post_quality || {};
  const comp = metrics?.comparison || {};

  // Chart data for dimensional comparison
  const chartData = [
    { name: 'Completeness', Pre: preQ.completeness || 0, Post: postQ.completeness || 0 },
    { name: 'Validity', Pre: preQ.validity || 0, Post: postQ.validity || 0 },
    { name: 'Uniqueness', Pre: preQ.uniqueness || 0, Post: postQ.uniqueness || 0 },
    { name: 'SNR / Integrity', Pre: preQ.snr || 0, Post: postQ.snr || 0 },
  ];

  return (
    <div className="space-y-6">
      {/* Executive Quality Scorecards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex items-center justify-between">
          <div>
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
              Baseline Score
            </span>
            <p className="text-3xl font-extrabold text-slate-300 mt-1">
              {pre_score.toFixed(1)} <span className="text-xs font-normal text-slate-500">/ 100</span>
            </p>
          </div>
          <div className="w-12 h-12 rounded-xl bg-slate-800 flex items-center justify-center text-slate-400">
            <Award className="w-6 h-6" />
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex items-center justify-between">
          <div>
            <span className="text-xs font-medium text-brand-400 uppercase tracking-wider">
              Optimized Score
            </span>
            <p className="text-3xl font-extrabold text-brand-400 mt-1">
              {post_score.toFixed(1)} <span className="text-xs font-normal text-slate-500">/ 100</span>
            </p>
          </div>
          <div className="w-12 h-12 rounded-xl bg-brand-500/10 border border-brand-500/20 flex items-center justify-center text-brand-400">
            <Award className="w-6 h-6" />
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex items-center justify-between">
          <div>
            <span className="text-xs font-medium text-emerald-400 uppercase tracking-wider">
              Quality Delta
            </span>
            <p className="text-3xl font-extrabold text-emerald-400 mt-1 flex items-center gap-1">
              <TrendingUp className="w-6 h-6" />
              {score_delta >= 0 ? `+${score_delta.toFixed(1)}` : score_delta.toFixed(1)}%
            </p>
          </div>
          <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
            <CheckCircle2 className="w-6 h-6" />
          </div>
        </div>
      </div>

      {/* Quality Dimensional Breakdown Chart */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
        <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
          Normalized Dimensional Quality Vectors (%)
        </h4>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <XAxis dataKey="name" stroke="#64748b" fontSize={11} tickLine={false} />
              <YAxis domain={[0, 100]} stroke="#64748b" fontSize={11} tickLine={false} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
              />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
              <Bar dataKey="Pre" fill="#475569" radius={[4, 4, 0, 0]} name="Baseline (Pre)" />
              <Bar dataKey="Post" fill="#22c55e" radius={[4, 4, 0, 0]} name="Optimized (Post)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Mathematical Formula & Individual Dimensional Contributions */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-slate-800">
          <div>
            <h4 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
              <Code className="w-4 h-4 text-brand-500" /> Transparent Quality Formula & Dimensional Contributions
            </h4>
            <p className="text-xs text-slate-400 mt-0.5">
              Quantified composite score derived from actual dataset metrics without arbitrary increases.
            </p>
          </div>
          <div className="font-mono text-xs bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 text-brand-400">
            {preQ.formula_display || 'Q = 35%·S_comp + 25%·S_val + 20%·S_uniq + 20%·S_snr'}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 px-3 font-semibold">Quality Dimension</th>
                <th className="py-2.5 px-3 font-semibold text-center">Weight (w)</th>
                <th className="py-2.5 px-3 font-semibold text-center">Baseline Metric (S_pre)</th>
                <th className="py-2.5 px-3 font-semibold text-center">Baseline Contrib</th>
                <th className="py-2.5 px-3 font-semibold text-center">Optimized Metric (S_post)</th>
                <th className="py-2.5 px-3 font-semibold text-center">Optimized Contrib</th>
                <th className="py-2.5 px-3 font-semibold text-right">Net Contribution Gain</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {[
                {
                  dim: 'Completeness (S_comp)',
                  desc: '1 - (missing / total cells)',
                  w: 35,
                  preMetric: preQ.completeness || 0,
                  preContrib: preQ.contributions?.completeness ?? ((preQ.completeness || 0) * 0.35),
                  postMetric: postQ.completeness || 0,
                  postContrib: postQ.contributions?.completeness ?? ((postQ.completeness || 0) * 0.35),
                },
                {
                  dim: 'Validity (S_val)',
                  desc: '1 - (invalid & corrupt cells / valid)',
                  w: 25,
                  preMetric: preQ.validity || 0,
                  preContrib: preQ.contributions?.validity ?? ((preQ.validity || 0) * 0.25),
                  postMetric: postQ.validity || 0,
                  postContrib: postQ.contributions?.validity ?? ((postQ.validity || 0) * 0.25),
                },
                {
                  dim: 'Uniqueness (S_uniq)',
                  desc: '1 - (duplicate rows / total rows)',
                  w: 20,
                  preMetric: preQ.uniqueness || 0,
                  preContrib: preQ.contributions?.uniqueness ?? ((preQ.uniqueness || 0) * 0.20),
                  postMetric: postQ.uniqueness || 0,
                  postContrib: postQ.contributions?.uniqueness ?? ((postQ.uniqueness || 0) * 0.20),
                },
                {
                  dim: 'Signal-to-Noise Ratio (S_snr)',
                  desc: '1 - (outliers / numeric cells)',
                  w: 20,
                  preMetric: preQ.snr || 0,
                  preContrib: preQ.contributions?.snr ?? ((preQ.snr || 0) * 0.20),
                  postMetric: postQ.snr || 0,
                  postContrib: postQ.contributions?.snr ?? ((postQ.snr || 0) * 0.20),
                },
              ].map((row) => {
                const diff = (row.postContrib - row.preContrib);
                return (
                  <tr key={row.dim} className="hover:bg-slate-950/40">
                    <td className="py-2.5 px-3 font-sans">
                      <p className="font-medium text-slate-200">{row.dim}</p>
                      <p className="text-[10px] text-slate-500 font-mono">{row.desc}</p>
                    </td>
                    <td className="py-2.5 px-3 text-center text-slate-400">{row.w}%</td>
                    <td className="py-2.5 px-3 text-center text-slate-300">{row.preMetric.toFixed(1)}%</td>
                    <td className="py-2.5 px-3 text-center text-slate-400">{row.preContrib.toFixed(2)} pts</td>
                    <td className="py-2.5 px-3 text-center text-brand-400 font-semibold">{row.postMetric.toFixed(1)}%</td>
                    <td className="py-2.5 px-3 text-center text-brand-400 font-semibold">{row.postContrib.toFixed(2)} pts</td>
                    <td className={`py-2.5 px-3 text-right font-semibold ${diff >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {diff >= 0 ? `+${diff.toFixed(2)}` : diff.toFixed(2)} pts
                    </td>
                  </tr>
                );
              })}
              <tr className="bg-slate-950 font-semibold text-white border-t border-slate-700">
                <td className="py-3 px-3 font-sans">Total Composite Score (Q)</td>
                <td className="py-3 px-3 text-center text-brand-400">100%</td>
                <td className="py-3 px-3 text-center">-</td>
                <td className="py-3 px-3 text-center text-slate-300">{pre_score.toFixed(1)} pts</td>
                <td className="py-3 px-3 text-center">-</td>
                <td className="py-3 px-3 text-center text-brand-400">{post_score.toFixed(1)} pts</td>
                <td className="py-3 px-3 text-right text-emerald-400">
                  {score_delta >= 0 ? `+${score_delta.toFixed(1)}` : score_delta.toFixed(1)} pts
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Column-Level Exact Imputation Strategy Audit */}
      {metrics?.imputation_strategies && Object.keys(metrics.imputation_strategies).length > 0 && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800">
            <div>
              <h4 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-500" /> Column-Level Imputation Strategy Audit
              </h4>
              <p className="text-xs text-slate-400 mt-0.5">
                Exact imputation strategy and resolution parameter applied to each column.
              </p>
            </div>
            <span className="text-xs text-slate-400">
              {Object.values(metrics.imputation_strategies).filter((v) => v.null_count > 0).length} columns imputed
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2.5 px-3 font-semibold">Column</th>
                  <th className="py-2.5 px-3 font-semibold">Data Type</th>
                  <th className="py-2.5 px-3 font-semibold">Null Count Before</th>
                  <th className="py-2.5 px-3 font-semibold">Exact Strategy Applied</th>
                  <th className="py-2.5 px-3 font-semibold">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {Object.entries(metrics.imputation_strategies).map(([colName, info]) => (
                  <tr key={colName} className="hover:bg-slate-950/40">
                    <td className="py-2.5 px-3 font-sans font-semibold text-slate-200">{colName}</td>
                    <td className="py-2.5 px-3 text-slate-400">{info.data_type}</td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {info.null_count} {info.null_count > 0 && <span className="text-[10px] text-slate-500">({info.null_percent}%)</span>}
                    </td>
                    <td className="py-2.5 px-3 text-brand-400 font-sans font-medium">
                      {info.description || info.strategy}
                    </td>
                    <td className="py-2.5 px-3">
                      <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full uppercase ${
                        info.null_count > 0
                          ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                          : 'bg-slate-800 text-slate-400'
                      }`}>
                        {info.null_count > 0 ? 'Resolved' : 'Clean'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Explainable Recommendations & Applied Diagnoses */}
      {recommendations.length > 0 && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <HelpCircle className="w-4 h-4 text-brand-500" /> Explainable Quality Diagnoses & Interventions
            </h4>
            <span className="text-xs text-slate-400">
              {recommendations.filter((r) => r.approved).length} of {recommendations.length} Approved
            </span>
          </div>

          <div className="space-y-3">
            {recommendations.map((rec) => (
              <div
                key={rec.id}
                className="bg-slate-950 p-4 rounded-xl border border-slate-800/80 flex items-start justify-between gap-4"
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-white capitalize">{rec.target || 'Dataset'}</span>
                    <span
                      className={`text-[10px] font-semibold px-2 py-0.5 rounded-full uppercase ${
                        rec.priority === 'critical' || rec.priority === 'high'
                          ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                          : 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                      }`}
                    >
                      {rec.priority || 'medium'}
                    </span>
                  </div>
                  <p className="text-xs text-slate-300">{rec.diagnosis}</p>
                  <p className="text-xs text-brand-400 font-medium">Applied: {rec.recommendation}</p>
                </div>

                <label className="flex items-center gap-2 cursor-pointer flex-shrink-0 text-xs text-slate-400">
                  <input
                    type="checkbox"
                    checked={rec.approved}
                    onChange={() => toggleRecommendation(rec.id)}
                    className="rounded border-slate-700 text-brand-500 focus:ring-0 bg-slate-900"
                  />
                  <span>Approved</span>
                </label>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Multi-Format Export & Artifact Downloads (Strictly dataset-type conditioned) */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
            <Download className="w-4 h-4 text-brand-500" /> Export Standardized Artifacts & Reports
          </h4>
          <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
            {isVision ? 'Visual Asset Pipeline' : 'Tabular Dataset Pipeline'}
          </span>
        </div>

        {isVision ? (
          /* Vision Dataset Exports: PNG, JPEG, ZIP, PDF, JSON (Never CSV/XLSX) */
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            <a
              href={api.getImageDownloadUrl(jobId, 'png')}
              download
              className="flex flex-col items-center justify-center p-3.5 rounded-xl bg-slate-950 border border-slate-800 hover:border-brand-500/50 hover:bg-slate-850 text-slate-200 transition group text-center"
            >
              <Download className="w-5 h-5 text-cyan-400 mb-1.5 group-hover:scale-110 transition" />
              <span className="text-xs font-medium">Final Image (PNG)</span>
              <span className="text-[10px] text-slate-500">Lossless Master</span>
            </a>

            <a
              href={api.getImageDownloadUrl(jobId, 'jpg')}
              download
              className="flex flex-col items-center justify-center p-3.5 rounded-xl bg-slate-950 border border-slate-800 hover:border-brand-500/50 hover:bg-slate-850 text-slate-200 transition group text-center"
            >
              <Download className="w-5 h-5 text-amber-400 mb-1.5 group-hover:scale-110 transition" />
              <span className="text-xs font-medium">Final Image (JPEG)</span>
              <span className="text-[10px] text-slate-500">95% High Quality</span>
            </a>

            <a
              href={api.getJobExportUrl(jobId, 'zip')}
              download
              className="flex flex-col items-center justify-center p-3.5 rounded-xl bg-slate-950 border border-slate-800 hover:border-brand-500/50 hover:bg-slate-850 text-slate-200 transition group text-center"
            >
              <Archive className="w-5 h-5 text-purple-400 mb-1.5 group-hover:scale-110 transition" />
              <span className="text-xs font-medium">Vision Bundle (ZIP)</span>
              <span className="text-[10px] text-slate-500">All Processed Images</span>
            </a>

            <a
              href={api.getJobExportUrl(jobId, 'pdf')}
              download
              className="flex flex-col items-center justify-center p-3.5 rounded-xl bg-slate-950 border border-slate-800 hover:border-brand-500/50 hover:bg-slate-850 text-slate-200 transition group text-center"
            >
              <FileText className="w-5 h-5 text-red-400 mb-1.5 group-hover:scale-110 transition" />
              <span className="text-xs font-medium">Audit Report (PDF)</span>
              <span className="text-[10px] text-slate-500">Quality & Benchmarks</span>
            </a>

            <a
              href={api.getJobExportUrl(jobId, 'json')}
              download
              className="flex flex-col items-center justify-center p-3.5 rounded-xl bg-slate-950 border border-slate-800 hover:border-brand-500/50 hover:bg-slate-850 text-slate-200 transition group text-center"
            >
              <Code className="w-5 h-5 text-slate-400 mb-1.5 group-hover:scale-110 transition" />
              <span className="text-xs font-medium">Run Manifest (JSON)</span>
              <span className="text-[10px] text-slate-500">Reproducibility Data</span>
            </a>
          </div>
        ) : (
          /* Tabular Dataset Exports: CSV, Excel, PDF, JSON (Never PNG/JPEG/ZIP) */
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <a
              href={api.getJobExportUrl(jobId, 'csv')}
              download
              className="flex flex-col items-center justify-center p-3.5 rounded-xl bg-slate-950 border border-slate-800 hover:border-brand-500/50 hover:bg-slate-850 text-slate-200 transition group text-center"
            >
              <FileSpreadsheet className="w-5 h-5 text-emerald-400 mb-1.5 group-hover:scale-110 transition" />
              <span className="text-xs font-medium">Cleaned CSV</span>
              <span className="text-[10px] text-slate-500">Standard RFC-4180</span>
            </a>

            <a
              href={api.getJobExportUrl(jobId, 'xlsx')}
              download
              className="flex flex-col items-center justify-center p-3.5 rounded-xl bg-slate-950 border border-slate-800 hover:border-brand-500/50 hover:bg-slate-850 text-slate-200 transition group text-center"
            >
              <FileSpreadsheet className="w-5 h-5 text-blue-400 mb-1.5 group-hover:scale-110 transition" />
              <span className="text-xs font-medium">Excel Workbook (.xlsx)</span>
              <span className="text-[10px] text-slate-500">Formatted Spreadsheet</span>
            </a>

            <a
              href={api.getJobExportUrl(jobId, 'pdf')}
              download
              className="flex flex-col items-center justify-center p-3.5 rounded-xl bg-slate-950 border border-slate-800 hover:border-brand-500/50 hover:bg-slate-850 text-slate-200 transition group text-center"
            >
              <FileText className="w-5 h-5 text-red-400 mb-1.5 group-hover:scale-110 transition" />
              <span className="text-xs font-medium">Audit Report (PDF)</span>
              <span className="text-[10px] text-slate-500">Quality & Imputation</span>
            </a>

            <a
              href={api.getJobExportUrl(jobId, 'json')}
              download
              className="flex flex-col items-center justify-center p-3.5 rounded-xl bg-slate-950 border border-slate-800 hover:border-brand-500/50 hover:bg-slate-850 text-slate-200 transition group text-center"
            >
              <Code className="w-5 h-5 text-purple-400 mb-1.5 group-hover:scale-110 transition" />
              <span className="text-xs font-medium">Run Manifest (JSON)</span>
              <span className="text-[10px] text-slate-500">Machine-Readable Run</span>
            </a>
          </div>
        )}
      </div>
    </div>
  );
}
