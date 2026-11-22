import React, { useState } from 'react';
import { ChevronLeft, ChevronRight, ArrowUpDown, Filter, Info, Hash, Type, Calendar, ToggleLeft } from 'lucide-react';

export default function TabularGrid({ previewData, onPageChange, loading }) {
  const [filterText, setFilterText] = useState('');
  const [selectedCol, setSelectedCol] = useState(null);
  const [sortCol, setSortCol] = useState(null);
  const [sortAsc, setSortAsc] = useState(true);

  if (!previewData || !previewData.rows) {
    return (
      <div className="p-8 text-center text-slate-500 bg-slate-900/40 rounded-xl border border-slate-800">
        No tabular preview data loaded.
      </div>
    );
  }

  const { columns, rows, total_rows, page, limit, total_pages, profile } = previewData;

  // Sorting and client filtering on current page rows
  let displayRows = [...rows];
  if (filterText) {
    displayRows = displayRows.filter((r) =>
      columns.some((c) => String(r[c] ?? '').toLowerCase().includes(filterText.toLowerCase()))
    );
  }

  if (sortCol) {
    displayRows.sort((a, b) => {
      const valA = a[sortCol];
      const valB = b[sortCol];
      if (valA === valB) return 0;
      if (valA === null || valA === undefined) return 1;
      if (valB === null || valB === undefined) return -1;
      return (valA > valB ? 1 : -1) * (sortAsc ? 1 : -1);
    });
  }

  const handleSort = (col) => {
    if (sortCol === col) {
      setSortAsc(!sortAsc);
    } else {
      setSortCol(col);
      setSortAsc(true);
    }
  };

  const getTypeIcon = (dtype) => {
    if (!dtype) return <Type className="w-3.5 h-3.5 text-slate-400" />;
    const dt = dtype.toLowerCase();
    if (dt.includes('int') || dt.includes('float')) return <Hash className="w-3.5 h-3.5 text-blue-400" />;
    if (dt.includes('bool')) return <ToggleLeft className="w-3.5 h-3.5 text-purple-400" />;
    if (dt.includes('date') || dt.includes('time')) return <Calendar className="w-3.5 h-3.5 text-amber-400" />;
    return <Type className="w-3.5 h-3.5 text-slate-400" />;
  };

  return (
    <div className="space-y-4">
      {/* Profiling Metrics Summary Cards */}
      {profile && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-3">
            <span className="text-xs text-slate-400 font-medium">Dimensions</span>
            <p className="text-lg font-semibold text-white mt-0.5">
              {profile.total_rows.toLocaleString()} <span className="text-xs font-normal text-slate-500">rows</span> × {profile.total_cols} <span className="text-xs font-normal text-slate-500">cols</span>
            </p>
          </div>
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-3">
            <span className="text-xs text-slate-400 font-medium">Memory Footprint</span>
            <p className="text-lg font-semibold text-white mt-0.5">
              {profile.memory_mb} <span className="text-xs font-normal text-slate-500">MB</span>
            </p>
          </div>
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-3">
            <span className="text-xs text-slate-400 font-medium">Duplicate Rows</span>
            <p className="text-lg font-semibold text-amber-400 mt-0.5">
              {profile.duplicate_rows.toLocaleString()}
            </p>
          </div>
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-3">
            <span className="text-xs text-slate-400 font-medium">Numeric Features</span>
            <p className="text-lg font-semibold text-brand-400 mt-0.5">
              {profile.numeric_columns ? profile.numeric_columns.length : 0}
            </p>
          </div>
        </div>
      )}

      {/* Grid Controls */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 bg-slate-900/80 p-3 rounded-xl border border-slate-800">
        <div className="relative w-full sm:w-72">
          <Filter className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
            placeholder="Filter current view rows..."
            className="w-full bg-slate-950 border border-slate-700 rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-brand-500"
          />
        </div>

        {/* Pagination Controls */}
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-400">
            Page <span className="font-semibold text-slate-200">{page}</span> of{' '}
            <span className="font-semibold text-slate-200">{total_pages}</span> ({total_rows.toLocaleString()} total)
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => onPageChange && onPageChange(page - 1)}
              disabled={page <= 1 || loading}
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-slate-300"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => onPageChange && onPageChange(page + 1)}
              disabled={page >= total_pages || loading}
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-slate-300"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Virtualized/Scrollable Data Table */}
      <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/50 shadow-inner max-h-[500px]">
        <table className="w-full text-left border-collapse text-xs">
          <thead className="bg-slate-900/90 sticky top-0 z-10 backdrop-blur border-b border-slate-800">
            <tr>
              <th className="p-3 font-semibold text-slate-400 border-r border-slate-800 w-12 text-center">#</th>
              {columns.map((col) => {
                const colMeta = profile?.columns?.[col];
                return (
                  <th
                    key={col}
                    onClick={() => handleSort(col)}
                    className="p-3 font-semibold text-slate-300 border-r border-slate-800 last:border-r-0 cursor-pointer hover:bg-slate-800/60 transition select-none min-w-[130px]"
                  >
                    <div className="flex items-center justify-between gap-1.5">
                      <div className="flex items-center gap-1.5 truncate">
                        {getTypeIcon(colMeta?.dtype)}
                        <span className="truncate">{col}</span>
                      </div>
                      <ArrowUpDown className="w-3 h-3 text-slate-500 flex-shrink-0" />
                    </div>
                    {colMeta && (
                      <div className="flex items-center gap-1.5 mt-1 text-[10px] font-normal text-slate-500">
                        <span>{colMeta.dtype}</span>
                        {colMeta.null_percent > 0 && (
                          <span className="text-amber-400 font-medium">({colMeta.null_percent}% null)</span>
                        )}
                      </div>
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-mono">
            {displayRows.length === 0 ? (
              <tr>
                <td colSpan={columns.length + 1} className="p-8 text-center text-slate-500">
                  No matching rows found.
                </td>
              </tr>
            ) : (
              displayRows.map((row, idx) => (
                <tr key={idx} className="hover:bg-slate-800/40 transition">
                  <td className="p-2.5 text-center text-slate-500 border-r border-slate-800/60">
                    {(page - 1) * limit + idx + 1}
                  </td>
                  {columns.map((col) => {
                    const val = row[col];
                    const isNull = val === null || val === undefined;
                    return (
                      <td
                        key={col}
                        className={`p-2.5 border-r border-slate-800/60 last:border-r-0 truncate max-w-[200px] ${
                          isNull ? 'bg-amber-500/10 text-amber-400 italic font-sans' : 'text-slate-300'
                        }`}
                      >
                        {isNull ? 'null' : String(val)}
                      </td>
                    );
                  })}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
