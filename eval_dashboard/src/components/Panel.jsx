import { useState, useEffect, useMemo, useCallback } from 'react'
import SearchBar from './SearchBar'
import ChipBar from './ChipBar'
import ExperimentList from './ExperimentList'
import IndexSummaryCard from './IndexSummaryCard'
import ExperimentTable, { DEFAULT_VISIBLE_COLUMNS, ALL_COLUMNS } from './ExperimentTable'
import { useIndexData } from '../hooks/useIndexData'
import { useExperiment } from '../hooks/useExperiment'

/**
 * Self-contained experiment panel (base or compare).
 *
 * Layout (flex-col, h-full):
 *   [1] Panel header        — shrink-0
 *   [2] Search + ChipBar    — shrink-0
 *   [3] Experiment list     — overflow-y-auto, maxHeight ~35%
 *   [4] IndexSummaryCard    — overflow-y-auto, maxHeight ~28%
 *   [5] ExperimentTable     — flex-1 min-h-0 (takes remaining space, own scroll)
 *   [6] Error banner        — shrink-0
 */
export default function Panel({ side, baseData, baseTableName, onExperimentLoad, onTableNameChange }) {
    const isCompare = side === 'compare'

    const { data: indexRows, loading: loadingIndex, error: indexError, refetch } = useIndexData()

    const [searchQuery, setSearchQuery] = useState('')
    const [columnFilters, setColumnFilters] = useState({})
    const [selectedRow, setSelectedRow] = useState(null)

    const { data: experimentData, loading: loadingExperiment, error: expError } =
        useExperiment(selectedRow?.table_name ?? null)

    useEffect(() => { onExperimentLoad(experimentData ?? null) }, [experimentData, onExperimentLoad])
    useEffect(() => { if (onTableNameChange) onTableNameChange(selectedRow?.table_name ?? null) }, [selectedRow, onTableNameChange])

    const [visibleColumns, setVisibleColumns] = useState(DEFAULT_VISIBLE_COLUMNS)

    const toggleColumn = useCallback((col) => {
        setVisibleColumns((prev) => {
            const next = prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
            // Always sort based on master ALL_COLUMNS order
            return ALL_COLUMNS.filter(c => next.includes(c))
        })
    }, [])

    const handleFilterChange = useCallback((col, values) => {
        setColumnFilters((prev) => {
            const next = { ...prev }
            if (values.length === 0) delete next[col]
            else next[col] = values
            return next
        })
    }, [])

    const filteredRows = useMemo(() => {
        if (!indexRows) return []
        if (selectedRow) return [selectedRow]

        let rows = indexRows
        const q = searchQuery.toLowerCase().trim()
        if (q) rows = rows.filter((r) => r.table_name.toLowerCase().includes(q))

        Object.entries(columnFilters).forEach(([col, values]) => {
            if (values.length > 0) rows = rows.filter((r) => values.includes(r[col]))
        })
        return rows
    }, [indexRows, searchQuery, columnFilters, selectedRow])

    const handleClear = () => {
        setSelectedRow(null)
        setSearchQuery('')
        setColumnFilters({})
        if (onTableNameChange) onTableNameChange(null)
    }

    return (
        /*
         * h-full + flex-col + overflow-hidden:
         *   The panel fills its parent completely. Children are stacked vertically.
         *   overflow-hidden prevents children from accidentally bleeding out.
         */
        <div className="flex flex-col h-full overflow-hidden bg-[var(--bg-panel)] transition-colors">

            {/* ── [1] Panel header ─────────────────────────────────────── */}
            <div className="shrink-0 flex items-center justify-between px-3.5 py-1.5 border-b border-[var(--border-main)] bg-[var(--bg-header)] z-10 shadow-sm">
                <div className="flex items-center gap-2">
                    <span className={`text-[10px] font-black uppercase tracking-widest px-2 py-0.5 rounded-md border ${isCompare
                            ? 'text-cyan-600 bg-cyan-50 border-cyan-100 dark:bg-cyan-950/30 dark:border-cyan-900/50 dark:text-cyan-400'
                            : 'text-amber-600 bg-amber-50 border-amber-100 dark:bg-amber-950/30 dark:border-amber-900/50 dark:text-amber-400'
                        }`}>
                        {isCompare ? 'Compare' : 'Base'}
                    </span>
                    <span className="text-[10px] text-[var(--text-muted)] font-black uppercase tracking-tight opacity-50">
                        Intelligence Layer
                    </span>
                </div>
                {selectedRow && (
                    <button
                        onClick={handleClear}
                        className="group flex items-center gap-1.5 px-2 py-1 text-[10px] text-red-500 font-black uppercase tracking-tighter hover:bg-red-50 dark:hover:bg-red-950/20 rounded-md transition-all focus:outline-none"
                    >
                        <svg className="h-3 w-3 transition-transform group-hover:rotate-180 duration-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                            <path d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m0 0H15" />
                        </svg>
                        Reset
                    </button>
                )}
            </div>

            {/* ── [2] Search + ChipBar ─────────────────────────────────── */}
            <div className="shrink-0 px-3 pt-2.5 pb-2 bg-[var(--bg-panel)] border-b border-[var(--border-main)] transition-colors">
                <SearchBar value={searchQuery} onChange={setSearchQuery} />
                <ChipBar
                    indexRows={indexRows}
                    columnFilters={columnFilters}
                    onFilterChange={handleFilterChange}
                />
            </div>

            {/* ── [3] Experiment list — scrollable section ─────────────── 
                max-height: 35% of panel height.
                If short list, it shrinks naturally; if long, user scrolls.
            ─────────────────────────────────────────────────────────────── */}
            <div
                className="border-b border-[var(--border-main)] overflow-y-auto"
                style={{ maxHeight: '35%', minHeight: '48px' }}
            >
                <ExperimentList
                    rows={filteredRows}
                    activeChips={Object.keys(columnFilters)}
                    selectedTableName={selectedRow?.table_name}
                    onSelect={setSelectedRow}
                    loading={loadingIndex}
                    error={indexError}
                />
                {indexError && (
                    <div className="px-3 pb-2">
                        <button
                            onClick={refetch}
                            className="text-xs text-[var(--accent)] hover:text-[var(--text-main)] underline underline-offset-2 transition-colors"
                        >
                            Retry Connection
                        </button>
                    </div>
                )}
            </div>

            {/* ── [4] Summary card — scrollable section ───────────────────
                When an experiment is selected this card shows metadata + KPIs.
                max-height: 28% so it never pushes the table off screen.
            ─────────────────────────────────────────────────────────────── */}
            <div
                className="overflow-y-auto shrink-0 border-b border-[var(--border-main)]"
                style={{ maxHeight: '28%' }}
            >
                <IndexSummaryCard
                    row={selectedRow}
                    side={side}
                    baseTableName={isCompare ? baseTableName : undefined}
                />
            </div>

            {/* ── [5] Experiment table — fills ALL remaining space ─────────
                flex-1 = grow to fill
                min-h-0 = CRITICAL: allows this flex child to shrink below its
                          content size so it doesn't push siblings out of the panel.
                overflow-hidden = the table manages its own internal scroll.
            ─────────────────────────────────────────────────────────────── */}
            <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
                <ExperimentTable
                    data={experimentData}
                    tableName={selectedRow?.table_name ?? null}
                    visibleColumns={visibleColumns}
                    onColumnToggle={toggleColumn}
                    baseData={isCompare ? baseData : null}
                    loading={loadingExperiment}
                />
            </div>

            {/* ── [6] Error banner ────────────────────────────────────────── */}
            {expError && (
                <div className="shrink-0 px-3 py-1.5 bg-red-900/10 border-t border-red-900/30">
                    <span className="text-xs text-red-500 font-medium">{expError}</span>
                </div>
            )}
        </div>
    )
}
