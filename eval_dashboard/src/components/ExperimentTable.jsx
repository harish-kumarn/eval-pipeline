import { useMemo, useRef, useState, useCallback } from 'react'
import { AgGridReact } from 'ag-grid-react'
import { useTheme } from '../context/ThemeContext'
import { formatBoolean, cn } from '../utils/formatUtils'
import { METRIC_COLUMNS } from '../utils/diffUtils'
import MetricBadge from './MetricBadge'
import PillList from './PillList'
import ColumnDropdown from './ColumnDropdown'
import ExportCSVButton from './ExportCSVButton'
import FilterPanel from './FilterPanel'
import { computeDiff } from '../utils/diffUtils'

const LIST_COLS = ['vector_chunk_ids', 'bm25_chunk_ids', 'pre_rerank_chunk_ids', 'post_rerank_chunk_ids', 'final_retrieved_chunks']
export const ALL_COLUMNS = [
    'id', 'timestamp', 'query', 'type', 'top_k',
    'reformulated_query',
    'vector_chunk_ids', 'bm25_chunk_ids', 'pre_rerank_chunk_ids', 'post_rerank_chunk_ids', 'final_retrieved_chunks',
    'actual_chunk',
    'mrr', 'recall', 'precision', 'ndcg',
    'faithfulness', 'context_precision', 'answer_correctness',
]

export const DEFAULT_VISIBLE_COLUMNS = [
    'query',
    'mrr', 'recall', 'ndcg', 'faithfulness', 'answer_correctness',
]

// ── Custom Cell Renderers ──────────────────────────────────────────

const MetricRenderer = ({ value, colDef, context }) => {
    const col = colDef.field
    const diff = context.diffMap?.get(context.rowIdMap?.get(context.data))?.get(col)
    const hasDiff = diff && diff.direction !== 'same'

    return (
        <span className={cn(
            'flex items-center h-full gap-1.5 px-0',
            hasDiff && diff.direction === 'up' && 'bg-[var(--diff-up-bg)] rounded px-1 -mx-1',
            hasDiff && diff.direction === 'down' && 'bg-[var(--diff-down-bg)] rounded px-1 -mx-1',
        )}>
            <MetricBadge value={value} />
            {hasDiff && (
                <span className={cn(
                    'text-[10px] font-mono font-black tabular-nums',
                    diff.direction === 'up' ? 'text-[var(--diff-up-text)]' : 'text-[var(--diff-down-text)]',
                )}>
                    {diff.direction === 'up' ? '▲' : '▼'}{Math.abs(diff.delta).toFixed(4)}
                </span>
            )}
        </span>
    )
}

const ListRenderer = ({ value }) => <PillList value={value} />
const BooleanRenderer = ({ value }) => <span className="text-xs font-mono font-medium text-[var(--text-main)]">{formatBoolean(value)}</span>
const TypeRenderer = ({ value }) => value ? (
    <span className="inline-block px-2 py-0.5 rounded-md text-[10px] font-black uppercase bg-[var(--bg-card)] text-[var(--text-muted)] border border-[var(--border-main)] tracking-widest shadow-sm">
        {value}
    </span>
) : <span className="text-[var(--text-muted)] opacity-50">—</span>

const QuestionRenderer = ({ value }) => (
    <div className="flex flex-col justify-center h-full min-h-[48px] py-2 px-1">
        <span className="text-[11px] leading-relaxed font-medium text-[var(--text-main)] line-clamp-3">
            {value || <span className="text-[var(--text-muted)] opacity-40 italic">No content</span>}
        </span>
    </div>
)



/**
 * Main experiment data table with AG Grid + custom FilterPanel.
 */
export default function ExperimentTable({
    data,
    tableName,
    visibleColumns,
    onColumnToggle,
    baseData,
    loading,
}) {
    const gridRef = useRef()
    const { theme } = useTheme()

    // ── Filtered data (driven by FilterPanel) ─────────────────────────
    const [filteredData, setFilteredData] = useState(null)

    const handleFilteredData = useCallback((rows) => {
        setFilteredData(rows)
    }, [])

    // Reset filtered data when experiment changes
    const effectiveData = filteredData ?? data

    // ── Diff computation ──────────────────────────────────────────────
    const rowIdMap = useMemo(() => {
        if (!data) return null
        const map = new Map()
        data.forEach(row => map.set(row, row.query))
        return map
    }, [data])

    const diffMap = useMemo(() => {
        if (!baseData || !data) return null
        return computeDiff ? computeDiff(baseData, data) : null
    }, [baseData, data])

    const columnDefs = useMemo(() => {
        const defs = [
            {
                headerName: '#',
                valueGetter: (params) => params.node.rowIndex + 1,
                width: 60,
                pinned: 'left',
                sortable: false,
                filter: false,
                cellClass: 'text-center text-[var(--text-muted)] font-mono font-medium border-r border-[var(--border-main)] transition-colors',
            }
        ]

        visibleColumns.forEach(col => {
            const def = {
                headerName: col.toUpperCase().replace(/_/g, ' '),
                field: col,
                filter: false,
                // Explicit layout mapping
                flex: col === 'query' || col === 'reformulated_query' || col === 'actual_chunk' ? 3 : 1,
                minWidth: col === 'query' || col === 'reformulated_query' ? 350 : 120,
            }

            if (METRIC_COLUMNS.includes(col)) {
                def.cellRenderer = MetricRenderer
                def.minWidth = 100
                def.flex = 0.8
            } else if (LIST_COLS.includes(col)) {
                def.cellRenderer = ListRenderer
                def.minWidth = 180
                def.flex = 2
            } else if (col === 'type') {
                def.cellRenderer = TypeRenderer
                def.minWidth = 100
            } else if (col === 'query' || col === 'reformulated_query' || col === 'actual_chunk') {
                def.cellRenderer = QuestionRenderer
                def.autoHeight = true
            }

            defs.push(def)
        })

        return defs
    }, [visibleColumns])

    const defaultColDef = useMemo(() => ({
        sortable: true,
        filter: false,
        resizable: true,
        headerClass: 'text-[9px] uppercase font-black tracking-[0.1em] text-[var(--text-muted)] bg-[var(--bg-header)] border-b border-[var(--border-main)] transition-colors opacity-80',
        cellClass: 'border-b border-[var(--border-main)] flex items-center transition-colors duration-200 py-2',
    }), [])

    const gridContext = useMemo(() => ({
        diffMap,
        rowIdMap,
    }), [diffMap, rowIdMap])

    // ── Row count display ─────────────────────────────────────────────
    const totalRows = data?.length ?? 0
    const shownRows = effectiveData?.length ?? 0
    const isFiltered = filteredData !== null && shownRows !== totalRows

    return (
        <div className="flex flex-col h-full overflow-hidden bg-[var(--bg-app)]">
            {/* ── Toolbar ─────────────────────────────────────────────── */}
            <div className="flex items-center justify-between gap-4 px-4 py-2 bg-[var(--bg-header)] shrink-0 border-b border-[var(--border-main)] shadow-sm z-10">
                <div className="flex items-center gap-3">
                    {baseData && tableName && (
                        <div className="flex items-center gap-2 px-2.5 py-1 bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border border-cyan-500/20 rounded-lg">
                            <span className="text-[10px] font-black uppercase tracking-widest animate-pulse">Comparison Active</span>
                            <div className="h-3 w-px bg-cyan-500/20" />
                            <span className="text-[10px] font-mono font-black italic opacity-70">Δ vs Baseline</span>
                        </div>
                    )}
                    {data && (
                        <span className="text-[10px] text-[var(--text-muted)] font-mono">
                            {isFiltered ? (
                                <><span className="text-[var(--accent)] font-bold">{shownRows}</span> / {totalRows} rows</>
                            ) : (
                                <>{totalRows} rows</>
                            )}
                        </span>
                    )}
                </div>
                <div className="flex items-center gap-2">
                    <ColumnDropdown
                        allColumns={ALL_COLUMNS}
                        visibleColumns={visibleColumns}
                        onToggle={onColumnToggle}
                    />
                    <div className="w-px h-4 bg-[var(--border-main)] mx-1" />
                    <ExportCSVButton tableName={tableName} />
                </div>
            </div>

            {/* ── Filter panel (only shown when data is loaded) ────────── */}
            {data && (
                <FilterPanel
                    columns={visibleColumns}
                    data={data}
                    onFilteredData={handleFilteredData}
                />
            )}

            {/* ── Grid Area ────────────────────────────────────────────── */}
            {loading ? (
                <div className="flex-1 flex flex-col items-center justify-center gap-4 bg-[var(--bg-app)] opacity-60">
                    <div className="relative">
                        <div className="w-12 h-12 rounded-full border-2 border-[var(--border-main)] border-t-[var(--accent)] animate-spin" />
                        <div className="absolute inset-0 flex items-center justify-center">
                            <div className="w-2 h-2 bg-[var(--accent)] rounded-full animate-ping" />
                        </div>
                    </div>
                    <div className="flex flex-col items-center gap-1">
                        <p className="text-[10px] text-[var(--text-main)] font-black uppercase tracking-[0.25em]">Hydrating Neural Layers</p>
                        <p className="text-[9px] text-[var(--text-muted)] italic">Synchronizing experiment data metrics…</p>
                    </div>
                </div>
            ) : !data ? (
                <div className="flex-1 flex flex-col items-center justify-center gap-5 bg-[var(--bg-app)] p-12">
                    <div className="w-16 h-16 rounded-2xl bg-[var(--bg-card)] border border-[var(--border-main)] flex items-center justify-center shadow-inner opacity-40">
                        <svg className="w-8 h-8 text-[var(--text-muted)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.628.251a2 2 0 01-1.268 0l-.628-.251a6 6 0 00-3.86-.517l-2.387.477a2 2 0 00-1.022.547l-.953 1.906A2 2 0 001.996 17.5V20a2 2 0 002 2h16a2 2 0 002-2v-2.5a2 2 0 00-1.572-1.906l-.953-1.906z" />
                        </svg>
                    </div>
                    <div className="flex flex-col items-center gap-2 max-w-[280px]">
                        <p className="text-[11px] text-[var(--text-main)] font-black uppercase tracking-[0.2em] text-center">Data Intelligence Offline</p>
                        <p className="text-[10px] text-[var(--text-muted)] text-center leading-relaxed opacity-70">
                            Select an experiment from the workspace above to reveal low-level telemetry and RAG performance insights.
                        </p>
                    </div>
                </div>
            ) : effectiveData && effectiveData.length === 0 ? (
                <div className="flex-1 flex flex-col items-center justify-center gap-3 bg-[var(--bg-app)] opacity-70">
                    <svg className="w-8 h-8 text-[var(--text-muted)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M3 4h18M7 10h10M11 16h2" />
                    </svg>
                    <p className="text-[11px] text-[var(--text-muted)] font-bold uppercase tracking-widest">No Rows Match</p>
                    <p className="text-[10px] text-[var(--text-muted)] opacity-60">Try adjusting or clearing your filters.</p>
                </div>
            ) : (
                <div className={cn(
                    "flex-1 min-h-0 w-full transition-colors",
                    theme === 'dark' ? 'ag-theme-balham-dark' : 'ag-theme-balham'
                )}>
                    <AgGridReact
                        ref={gridRef}
                        rowData={effectiveData}
                        columnDefs={columnDefs}
                        defaultColDef={{
                            ...defaultColDef,
                            cellClass: cn(
                                defaultColDef.cellClass,
                                'px-3 transition-colors duration-200'
                            )
                        }}
                        context={gridContext}
                        animateRows={true}
                        rowSelection="single"
                        pagination={false}
                        headerHeight={36}
                        autoSizeStrategy={{ type: 'fitCellContents' }}
                    />
                </div>
            )}
        </div>
    )
}
