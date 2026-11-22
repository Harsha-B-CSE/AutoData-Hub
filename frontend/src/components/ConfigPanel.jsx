import React, { useState, useEffect } from 'react';
import {
  Sliders,
  Cpu,
  Play,
  Save,
  Check,
  RefreshCw,
  HelpCircle,
  ChevronDown,
  ChevronUp,
  ShieldAlert,
  Sparkles,
  Info
} from 'lucide-react';
import { api } from '../services/api';

/**
 * Clean, lightweight Tooltip component explaining technical concepts in everyday language.
 */
function SettingTooltip({ text }) {
  const [show, setShow] = useState(false);

  return (
    <div className="relative inline-flex items-center ml-1.5 align-middle">
      <button
        type="button"
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        onClick={() => setShow(!show)}
        className="text-slate-500 hover:text-slate-300 transition-colors focus:outline-none"
        aria-label="Information"
      >
        <HelpCircle className="w-3.5 h-3.5" />
      </button>
      {show && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-2.5 bg-slate-900 border border-slate-700/80 rounded-xl shadow-2xl text-[11px] text-slate-200 z-50 pointer-events-none leading-relaxed backdrop-blur-md">
          {text}
          <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1 border-4 border-transparent border-t-slate-700/80" />
        </div>
      )}
    </div>
  );
}

export default function ConfigPanel({
  engineType = 'tabular',
  datasetId,
  previewData = null,
  onJobDispatched,
  disabled = false,
}) {
  const [presets, setPresets] = useState({ built_in: [], custom: [] });
  const [selectedPresetId, setSelectedPresetId] = useState('');
  const [executionMode, setExecutionMode] = useState('batch');
  const [loading, setLoading] = useState(false);
  const [saveName, setSaveName] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Tabular State (with sensible recommended defaults)
  const [tabConfig, setTabConfig] = useState({
    cleaning: {
      remove_duplicates: true,
      duplicate_keep: 'first',
      clean_text: true,
      text_case: 'preserve',
      infer_types: true,
      handle_outliers: true,
      outlier_method: 'iqr',
      outlier_action: 'clip',
      outlier_threshold: 1.5,
    },
    imputation: {
      numerical_strategy: 'median',
      categorical_strategy: 'mode',
      num_constant: 0.0,
      cat_constant: 'MISSING',
      knn_neighbors: 5,
    },
    scaling: {
      method: 'standard',
      feature_range: [0, 1],
    },
    encoding: {
      method: 'onehot',
      columns: null,
      exclude_columns: [],
      drop_first: false,
      max_categories: 25,
    },
  });

  // Vision State (with sensible recommended defaults: Adaptive Balanced, RGB color preserved)
  const [visConfig, setVisConfig] = useState({
    transform: {
      resize: true,
      width: 512,
      height: 512,
      preserve_aspect_ratio: true,
      interpolation: 'bilinear',
      rotation_angle: 0,
      flip_mode: '',
    },
    filter: {
      color_space: 'RGB',
      adaptive: true,
      adaptive_preset: 'balanced',
      filter_type: 'bilateral',
      kernel_size: 5,
      contrast_enhancement: 'clahe',
      clahe_clip_limit: 2.0,
    },
    augmentation: {
      enabled: false,
      horizontal_flip: true,
      vertical_flip: false,
      color_jitter: false,
      gaussian_noise: false,
    },
  });

  const profileCols = previewData?.profile?.columns || {};
  const allColumns = previewData?.columns || Object.keys(profileCols);
  const categoricalColumns = allColumns.filter(
    (col) => (profileCols[col] ? !profileCols[col].is_numeric : true)
  );

  const isIdentOrName = (col) => {
    if (profileCols[col]?.is_identifier_or_name !== undefined) {
      return profileCols[col].is_identifier_or_name;
    }
    const c = String(col).toLowerCase();
    return /((^|_)(id|uuid|guid|pk|fk|key|name|full_name|first_name|last_name|email|phone|ssn)($|_))/.test(c);
  };

  const handleToggleCol = (col, checked) => {
    const currentCols = tabConfig.encoding.columns !== null
      ? [...tabConfig.encoding.columns]
      : categoricalColumns.filter((c) => !isIdentOrName(c));

    let updated;
    if (checked) {
      updated = Array.from(new Set([...currentCols, col]));
    } else {
      updated = currentCols.filter((c) => c !== col);
    }
    const excluded = categoricalColumns.filter((c) => !updated.includes(c));
    setTabConfig({
      ...tabConfig,
      encoding: {
        ...tabConfig.encoding,
        columns: updated,
        exclude_columns: excluded,
      },
    });
  };

  const handleResetEncodingCols = () => {
    const defaultCols = categoricalColumns.filter((c) => !isIdentOrName(c));
    const excluded = categoricalColumns.filter((c) => isIdentOrName(c));
    setTabConfig({
      ...tabConfig,
      encoding: {
        ...tabConfig.encoding,
        columns: defaultCols,
        exclude_columns: excluded,
      },
    });
  };

  const handleSelectAllEncodingCols = () => {
    setTabConfig({
      ...tabConfig,
      encoding: {
        ...tabConfig.encoding,
        columns: categoricalColumns,
        exclude_columns: [],
      },
    });
  };

  useEffect(() => {
    loadPresets();
  }, []);

  const loadPresets = async () => {
    try {
      const data = await api.getConfigPresets();
      setPresets(data);
      const match = data.built_in.find((p) => p.engine_type === engineType);
      if (match) {
        setSelectedPresetId(match.id);
        applyPreset(match);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const applyPreset = (preset) => {
    if (preset.engine_type === 'tabular') {
      setTabConfig((prev) => ({ ...prev, ...preset.parameters }));
    } else if (preset.engine_type === 'vision') {
      setVisConfig((prev) => ({ ...prev, ...preset.parameters }));
    }
  };

  const handlePresetSelect = (id) => {
    setSelectedPresetId(id);
    const all = [...presets.built_in, ...presets.custom];
    const found = all.find((p) => p.id === id);
    if (found) {
      applyPreset(found);
    }
  };

  const handleSavePreset = async () => {
    if (!saveName.trim()) return;
    try {
      setIsSaving(true);
      const params = engineType === 'tabular' ? tabConfig : visConfig;
      await api.savePreset(saveName, engineType, params);
      setSaveName('');
      await loadPresets();
    } catch (err) {
      alert(err.message);
    } finally {
      setIsSaving(false);
    }
  };

  const handleDispatch = async () => {
    if (!datasetId) {
      alert('Please select or upload a dataset first.');
      return;
    }
    try {
      setLoading(true);
      const params = engineType === 'tabular' ? tabConfig : visConfig;
      const res = await api.dispatchJob(datasetId, selectedPresetId, params, executionMode);
      if (onJobDispatched) {
        onJobDispatched(res.job_id);
      }
    } catch (err) {
      alert(err.message || 'Failed to dispatch pipeline.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800/80 rounded-2xl p-6 space-y-6 shadow-xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <Sliders className="w-5 h-5 text-brand-400" />
            <h3 className="text-base font-semibold text-white tracking-tight">
              Pipeline Configuration
            </h3>
            <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-full bg-brand-500/10 text-brand-400 border border-brand-500/20 font-bold">
              {engineType} engine
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Choose a preset or customize how your data is cleaned, enhanced, and transformed.
          </p>
        </div>

        {/* Preset Selector */}
        <div className="flex items-center gap-2">
          <select
            value={selectedPresetId}
            onChange={(e) => handlePresetSelect(e.target.value)}
            className="bg-slate-950 border border-slate-700 text-xs text-slate-200 rounded-xl px-3 py-2 focus:outline-none focus:border-brand-500 shadow-sm"
          >
            <optgroup label="Recommended Presets">
              {presets.built_in
                .filter((p) => p.engine_type === engineType)
                .map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
            </optgroup>
            {presets.custom.filter((p) => p.engine_type === engineType).length > 0 && (
              <optgroup label="Custom Saved Presets">
                {presets.custom
                  .filter((p) => p.engine_type === engineType)
                  .map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
              </optgroup>
            )}
          </select>
        </div>
      </div>

      {/* ===================================================================== */}
      {/* Tabular Engine Configuration */}
      {/* ===================================================================== */}
      {engineType === 'tabular' && (
        <div className="space-y-5">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* 1. Duplicate Removal & Cleaning */}
            <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center">
                  Duplicate Rows & Cleaning
                  <SettingTooltip text="Deletes exact duplicate records so your dataset doesn't have artificial bias or repeated entries." />
                </span>
                <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20 font-mono">
                  Recommended
                </span>
              </div>

              <div className="space-y-2.5 text-xs">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={tabConfig.cleaning.remove_duplicates}
                    onChange={(e) =>
                      setTabConfig({
                        ...tabConfig,
                        cleaning: { ...tabConfig.cleaning, remove_duplicates: e.target.checked },
                      })
                    }
                    className="rounded border-slate-700 text-brand-500 focus:ring-0 bg-slate-900"
                  />
                  <span className="text-slate-300">Remove exact duplicate rows</span>
                </label>

                {tabConfig.cleaning.remove_duplicates && (
                  <div className="pl-6 pt-1">
                    <span className="block text-slate-400 mb-1">Which duplicate to keep?</span>
                    <select
                      value={tabConfig.cleaning.duplicate_keep}
                      onChange={(e) =>
                        setTabConfig({
                          ...tabConfig,
                          cleaning: { ...tabConfig.cleaning, duplicate_keep: e.target.value },
                        })
                      }
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-slate-200"
                    >
                      <option value="first">Keep first occurrence (Standard)</option>
                      <option value="last">Keep last occurrence</option>
                      <option value="none">Drop all occurrences</option>
                    </select>
                  </div>
                )}
              </div>
            </div>

            {/* 2. Missing Value Imputation */}
            <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center">
                  Missing Value Treatment
                  <SettingTooltip text="Replaces empty/blank cells with statistically sound values (like median or most frequent category) so algorithms don't fail." />
                </span>
                <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20 font-mono">
                  Recommended
                </span>
              </div>

              <div className="space-y-2.5 text-xs">
                <div>
                  <span className="block text-slate-400 mb-1">Numbers Strategy</span>
                  <select
                    value={tabConfig.imputation.numerical_strategy}
                    onChange={(e) =>
                      setTabConfig({
                        ...tabConfig,
                        imputation: { ...tabConfig.imputation, numerical_strategy: e.target.value },
                      })
                    }
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-slate-200"
                  >
                    <option value="median">Median (Recommended - resilient to extreme values)</option>
                    <option value="mean">Mean (Average value)</option>
                    <option value="constant">Fixed Constant Number (0.0)</option>
                    <option value="none">Do Not Impute</option>
                  </select>
                </div>

                <div>
                  <span className="block text-slate-400 mb-1">Text/Category Strategy</span>
                  <select
                    value={tabConfig.imputation.categorical_strategy}
                    onChange={(e) =>
                      setTabConfig({
                        ...tabConfig,
                        imputation: { ...tabConfig.imputation, categorical_strategy: e.target.value },
                      })
                    }
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-slate-200"
                  >
                    <option value="mode">Most Frequent Category (Recommended)</option>
                    <option value="constant">Fixed Text Label ("MISSING")</option>
                    <option value="none">Do Not Impute</option>
                  </select>
                </div>
              </div>
            </div>

            {/* 3. Number Scaling */}
            <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center">
                  Number Rescaling (Standardization)
                  <SettingTooltip text="Standardizes numeric ranges (mean=0, std=1) so huge numbers like Income ($100k) don't overpower small numbers like Age (25)." />
                </span>
              </div>

              <div className="space-y-2 text-xs">
                <div>
                  <span className="block text-slate-400 mb-1">Scaling Method</span>
                  <select
                    value={tabConfig.scaling.method}
                    onChange={(e) =>
                      setTabConfig({
                        ...tabConfig,
                        scaling: { ...tabConfig.scaling, method: e.target.value },
                      })
                    }
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-slate-200"
                  >
                    <option value="standard">StandardScaler (Zero mean, unit variance - Recommended)</option>
                    <option value="minmax">MinMaxScaler (Rescale strictly between 0 and 1)</option>
                    <option value="none">Keep Original Raw Values</option>
                  </select>
                </div>
              </div>
            </div>

            {/* 4. Extreme Values (Outliers) */}
            <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center">
                  Extreme Value Smoothing (Outliers)
                  <SettingTooltip text="Identifies unrealistic values using Interquartile Range (IQR) and smoothly caps them to the boundary." />
                </span>
              </div>

              <div className="space-y-2.5 text-xs">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={tabConfig.cleaning.handle_outliers}
                    onChange={(e) =>
                      setTabConfig({
                        ...tabConfig,
                        cleaning: { ...tabConfig.cleaning, handle_outliers: e.target.checked },
                      })
                    }
                    className="rounded border-slate-700 text-brand-500 focus:ring-0 bg-slate-900"
                  />
                  <span className="text-slate-300">Cap extreme outliers (Clip to 1.5× IQR boundary)</span>
                </label>
              </div>
            </div>
          </div>

          {/* Categorical Column Selection for One-Hot Encoding */}
          <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80 space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center">
                  Categorical Column Encoding (One-Hot)
                  <SettingTooltip text="Converts text categories into binary 0/1 indicator columns. Name and Identifier columns are safely excluded by default to avoid memory bloat." />
                </span>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  Identifiers & Name columns are automatically excluded by default. You can manually check or uncheck any column below.
                </p>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleResetEncodingCols}
                  className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[11px] text-slate-300 transition"
                  title="Reset to recommended defaults (exclude Name/ID)"
                >
                  Reset Defaults
                </button>
                <button
                  type="button"
                  onClick={handleSelectAllEncodingCols}
                  className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[11px] text-slate-300 transition"
                >
                  Select All
                </button>
              </div>
            </div>

            {categoricalColumns.length > 0 ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 pt-2">
                {categoricalColumns.map((col) => {
                  const isId = isIdentOrName(col);
                  const isSelected = tabConfig.encoding.columns !== null
                    ? tabConfig.encoding.columns.includes(col)
                    : !isId;

                  return (
                    <label
                      key={col}
                      className={`flex items-center gap-2 p-2 rounded-lg border text-xs cursor-pointer transition ${
                        isSelected
                          ? 'bg-brand-500/10 border-brand-500/40 text-brand-300'
                          : 'bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-700'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={(e) => handleToggleCol(col, e.target.checked)}
                        className="rounded border-slate-700 text-brand-500 focus:ring-0 bg-slate-950"
                      />
                      <span className="truncate font-mono">{col}</span>
                      {isId && (
                        <span className="text-[9px] px-1 py-0.2 rounded bg-amber-500/15 text-amber-400 border border-amber-500/30 ml-auto shrink-0 font-sans">
                          ID
                        </span>
                      )}
                    </label>
                  );
                })}
              </div>
            ) : (
              <p className="text-xs text-slate-500 italic">No non-numeric categorical columns detected in this dataset.</p>
            )}
          </div>
        </div>
      )}

      {/* ===================================================================== */}
      {/* Vision Engine Configuration */}
      {/* ===================================================================== */}
      {engineType === 'vision' && (
        <div className="space-y-5">
          {/* Primary: Adaptive Image Enhancement Engine */}
          <div className="bg-slate-950/70 p-5 rounded-2xl border border-brand-500/30 space-y-4 shadow-lg">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
              <div>
                <span className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-brand-400" />
                  Adaptive Image Enhancement Engine
                  <SettingTooltip text="Automatically analyzes brightness, contrast, noise, sharpness and resolution before processing. Selects Gamma Correction for low-light, CLAHE for flat contrast, Non-Local Means for noise, and conservative unsharp masking. Skips unnecessary operations when the image is already good." />
                </span>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  Dynamic multi-factor intelligence tailored to each photograph's individual characteristics.
                </p>
              </div>

              <span className="text-[10px] font-mono uppercase px-2.5 py-1 rounded-full bg-brand-500/10 text-brand-300 border border-brand-500/30 shrink-0 font-bold">
                Adaptive Intelligence
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {[
                {
                  id: 'natural',
                  name: 'Natural Preset',
                  desc: 'High-fidelity subtle adjustments, zero halos, and texture preservation. Gentle shadow lift and minimal sharpening.',
                },
                {
                  id: 'balanced',
                  name: 'Balanced Preset (Default)',
                  desc: 'Harmonious enhancement: low-light shadow recovery, smart LAB contrast, edge-preserving NLM denoising, and crisp focus.',
                },
                {
                  id: 'strong',
                  name: 'Strong Preset',
                  desc: 'Firm shadow lift, higher dynamic contrast, and deeper noise suppression for challenging night or heavy grain captures.',
                },
              ].map((p) => {
                const isSelected = (visConfig.filter.adaptive_preset || 'balanced') === p.id && visConfig.filter.adaptive !== false;
                return (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() =>
                      setVisConfig({
                        ...visConfig,
                        filter: {
                          ...visConfig.filter,
                          adaptive: true,
                          adaptive_preset: p.id,
                        },
                      })
                    }
                    className={`p-3.5 rounded-xl border text-left transition relative flex flex-col justify-between ${
                      isSelected
                        ? 'bg-brand-500/10 border-brand-500 text-white shadow-md'
                        : 'bg-slate-900 border-slate-800 text-slate-300 hover:border-slate-700'
                    }`}
                  >
                    <div>
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold">{p.name}</span>
                        {isSelected && <span className="w-2 h-2 rounded-full bg-brand-400 animate-pulse" />}
                      </div>
                      <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">{p.desc}</p>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* 1. Color Fidelity & Space */}
            <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center">
                  Color Mode & Fidelity
                  <SettingTooltip text="Full Color RGB preserves the natural golden, warm, and true architecture colors without channel inversions. Grayscale converts to single-channel monochrome." />
                </span>
                <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20 font-mono">
                  Preserves Colors
                </span>
              </div>

              <div>
                <span className="block text-slate-400 text-xs mb-1">Color Format</span>
                <select
                  value={visConfig.filter.color_space}
                  onChange={(e) =>
                    setVisConfig({
                      ...visConfig,
                      filter: { ...visConfig.filter, color_space: e.target.value },
                    })
                  }
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-slate-200"
                >
                  <option value="RGB">Full Color RGB (Preserves original natural colors - Recommended)</option>
                  <option value="GRAY">Grayscale (Black & White, 1 Channel)</option>
                  <option value="HSV">HSV (Hue-Saturation-Value)</option>
                </select>
              </div>
            </div>

            {/* 2. Resizing & Shape Preservation */}
            <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center">
                  Resizing & Shape Preservation
                  <SettingTooltip text="Letterbox padding adds clean borders so non-square images (e.g. 16:9 panoramic palaces) don't get stretched or squashed into a square." />
                </span>
                <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20 font-mono">
                  No Distortion
                </span>
              </div>

              <div className="space-y-2 text-xs">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={visConfig.transform.preserve_aspect_ratio}
                    onChange={(e) =>
                      setVisConfig({
                        ...visConfig,
                        transform: { ...visConfig.transform, preserve_aspect_ratio: e.target.checked },
                      })
                    }
                    className="rounded border-slate-700 text-brand-500 focus:ring-0 bg-slate-900"
                  />
                  <span className="text-slate-300">Preserve Original Shape (Add Letterbox Padding)</span>
                </label>

                <div>
                  <span className="block text-slate-400 mb-1">Resizing Algorithm (Interpolation)</span>
                  <select
                    value={visConfig.transform.interpolation}
                    onChange={(e) =>
                      setVisConfig({
                        ...visConfig,
                        transform: { ...visConfig.transform, interpolation: e.target.value },
                      })
                    }
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-slate-200"
                  >
                    <option value="bilinear">Bilinear Interpolation (Standard - Fast & balanced)</option>
                    <option value="bicubic">Bicubic Interpolation (Smooth curves, great for portraits)</option>
                    <option value="lanczos">Lanczos 8x8 (Ultra-sharp detail preservation)</option>
                  </select>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ===================================================================== */}
      {/* Optional Collapsible Advanced Settings Section */}
      {/* ===================================================================== */}
      <div className="border border-slate-800/80 rounded-xl overflow-hidden bg-slate-950/40">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full p-3.5 flex items-center justify-between text-xs font-semibold text-slate-300 hover:text-white bg-slate-900/40 hover:bg-slate-900/80 transition"
        >
          <span className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-brand-400" />
            <span>Advanced Settings & Hardware Allocation</span>
          </span>
          <span className="flex items-center gap-1.5 text-slate-400 font-mono text-[11px]">
            {showAdvanced ? 'Hide' : 'Show'}
            {showAdvanced ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </span>
        </button>

        {showAdvanced && (
          <div className="p-4 space-y-4 border-t border-slate-800 text-xs">
            {/* Hardware Paradigm */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <span className="font-semibold text-white">Execution Scheduling Mode:</span>
                <p className="text-[11px] text-slate-400">
                  Parallel activates multi-core ProcessPoolExecutor for high-throughput batch workloads.
                </p>
              </div>

              <div className="flex items-center gap-1.5 bg-slate-900 p-1 rounded-xl border border-slate-800">
                {['sequential', 'batch', 'parallel'].map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => setExecutionMode(mode)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium uppercase font-mono transition ${
                      executionMode === mode
                        ? 'bg-brand-500 text-white shadow-sm'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {mode}
                  </button>
                ))}
              </div>
            </div>

            {/* Vision-Specific Dimensions */}
            {engineType === 'vision' && (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 pt-2 border-t border-slate-800/60">
                <div>
                  <span className="block text-slate-400 mb-1">Target Width (px)</span>
                  <input
                    type="number"
                    value={visConfig.transform.width}
                    onChange={(e) =>
                      setVisConfig({
                        ...visConfig,
                        transform: { ...visConfig.transform, width: parseInt(e.target.value) || 512 },
                      })
                    }
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-slate-200 font-mono"
                  />
                </div>
                <div>
                  <span className="block text-slate-400 mb-1">Target Height (px)</span>
                  <input
                    type="number"
                    value={visConfig.transform.height}
                    onChange={(e) =>
                      setVisConfig({
                        ...visConfig,
                        transform: { ...visConfig.transform, height: parseInt(e.target.value) || 512 },
                      })
                    }
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-slate-200 font-mono"
                  />
                </div>
                <div>
                  <span className="block text-slate-400 mb-1">Denoise Kernel Size</span>
                  <select
                    value={visConfig.filter.kernel_size}
                    onChange={(e) =>
                      setVisConfig({
                        ...visConfig,
                        filter: { ...visConfig.filter, kernel_size: parseInt(e.target.value) || 5 },
                      })
                    }
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-slate-200 font-mono"
                  >
                    <option value="3">3×3 (Fine)</option>
                    <option value="5">5×5 (Standard Balanced)</option>
                    <option value="7">7×7 (Strong Denoising)</option>
                  </select>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Action Footer */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-3 border-t border-slate-800/80">
        {/* Preset Save */}
        <div className="flex items-center gap-2 w-full sm:w-auto">
          <input
            type="text"
            value={saveName}
            onChange={(e) => setSaveName(e.target.value)}
            placeholder="Save configuration as preset..."
            className="bg-slate-950 border border-slate-700 text-xs text-slate-200 rounded-xl px-3 py-2 w-full sm:w-60 focus:outline-none focus:border-brand-500 shadow-sm"
          />
          <button
            onClick={handleSavePreset}
            disabled={!saveName.trim() || isSaving}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-xs font-medium text-slate-200 transition"
          >
            <Save className="w-3.5 h-3.5" /> Save
          </button>
        </div>

        {/* Primary Run Button */}
        <button
          onClick={handleDispatch}
          disabled={disabled || loading || !datasetId}
          className="w-full sm:w-auto flex items-center justify-center gap-2 px-7 py-2.5 rounded-xl bg-brand-500 hover:bg-brand-600 active:scale-98 disabled:opacity-40 disabled:cursor-not-allowed text-white font-medium text-xs transition shadow-lg shadow-brand-500/25"
        >
          {loading ? (
            <RefreshCw className="w-4 h-4 animate-spin" />
          ) : (
            <Play className="w-4 h-4 fill-current" />
          )}
          <span>{loading ? 'Dispatching Pipeline...' : 'Execute Preprocessing Pipeline'}</span>
        </button>
      </div>
    </div>
  );
}
