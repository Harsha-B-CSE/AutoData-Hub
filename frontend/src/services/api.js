const API_BASE = '/api/v1';

export const api = {
  // System Health
  async getHealth() {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Failed to fetch system health');
    return res.json();
  },

  // Datasets & Ingestion
  async listDatasets() {
    const res = await fetch(`${API_BASE}/datasets`);
    if (!res.ok) throw new Error('Failed to list datasets');
    return res.json();
  },

  async uploadDataset(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Upload failed');
    }
    return res.json();
  },

  async previewDataset(datasetId, page = 1, limit = 50) {
    const res = await fetch(`${API_BASE}/datasets/${datasetId}/preview?page=${page}&limit=${limit}`);
    if (!res.ok) throw new Error('Failed to load dataset preview');
    return res.json();
  },

  getDatasetImageUrl(datasetId, imageName) {
    return `${API_BASE}/datasets/${datasetId}/image/${encodeURIComponent(imageName)}`;
  },

  // Configuration Presets
  async getConfigPresets() {
    const res = await fetch(`${API_BASE}/config/presets`);
    if (!res.ok) throw new Error('Failed to fetch config presets');
    return res.json();
  },

  async savePreset(name, engineType, parameters) {
    const res = await fetch(`${API_BASE}/config/presets`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, engine_type: engineType, parameters }),
    });
    if (!res.ok) throw new Error('Failed to save preset');
    return res.json();
  },

  // Job Dispatch & Monitoring
  async dispatchJob(datasetId, configId, parameters = null, executionMode = 'batch') {
    const res = await fetch(`${API_BASE}/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_id: datasetId,
        config_id: configId,
        parameters,
        execution_mode: executionMode,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to dispatch processing job');
    }
    return res.json();
  },

  async listJobs() {
    const res = await fetch(`${API_BASE}/jobs`);
    if (!res.ok) throw new Error('Failed to list jobs');
    return res.json();
  },

  async getJobStatus(jobId) {
    const res = await fetch(`${API_BASE}/jobs/${jobId}`);
    if (!res.ok) throw new Error('Failed to get job status');
    return res.json();
  },

  async getJobBenchmarks(jobId) {
    const res = await fetch(`${API_BASE}/jobs/${jobId}/benchmarks`);
    if (!res.ok) throw new Error('Failed to get benchmarks');
    return res.json();
  },

  async getJobQuality(jobId) {
    const res = await fetch(`${API_BASE}/jobs/${jobId}/quality`);
    if (!res.ok) throw new Error('Failed to get quality report');
    return res.json();
  },

  async getJobDiff(jobId, limit = 25) {
    const res = await fetch(`${API_BASE}/jobs/${jobId}/diff?limit=${limit}`);
    if (!res.ok) throw new Error('Failed to get job diff');
    return res.json();
  },

  getJobExportUrl(jobId, format = 'csv') {
    return `${API_BASE}/jobs/${jobId}/export?format=${format}`;
  },

  getImageDownloadUrl(jobId, format = 'png', imageName = null) {
    let url = `${API_BASE}/jobs/${jobId}/download-image?format=${format}`;
    if (imageName) {
      url += `&image_name=${encodeURIComponent(imageName)}`;
    }
    return url;
  },

  getProcessedImageUrl(jobId, imageName) {
    return `${API_BASE}/jobs/${jobId}/image/${encodeURIComponent(imageName)}`;
  },
};
