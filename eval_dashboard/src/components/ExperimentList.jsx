import { useMemo, useRef, useEffect } from 'react'
import { AgGridReact } from 'ag-grid-react'
import { useTheme } from '../context/ThemeContext'
import { formatMetric, formatBoolean, formatTimestamp, cn } from '../utils/formatUtils'
import MetricBadge from './MetricBadge'

const BOOL_KEYS = ['hybrid_search', 'query_reformulation', 'reranking']
const METRIC_KEYS = [
    'mrr_average', 'recall_average', 'precision_average', 'ndcg_average',
    'faithfulness_average', 'context_precision_average', 'answer_correctness_average',
]

/**
 * AG Grid based list of index rows (experiments).
 */
export default function ExperimentList({ rows, activeChips, selectedTableName, onSelect, loading, error }) {
    const gridRef = useRef()
    // ── ALL hooks MUST be called unconditionally at the top level ──
    const { theme } = useTheme()

    const columnDefs = useMemo(() => {
        const defs = [
            {
                headerName: 'Experiment',
                field: 'table_name',
                flex: 2,
                minWidth: 150,
                cellClass: 'font-mono text-xs font-bold text-[var(--text-main)]',
            }
        ]

        activeChips.forEach(chip => {
            const def = {
                headerName: chip.replace(/_average$/, '').toUpperCase(),
                field: chip,
                flex: 1,
                minWidth: 100,
            }

            if (chip === 'timestamp') {
                def.valueFormatter = (params) => formatTimestamp(params.value)
                def.cellClass = 'text-[var(--text-muted)] text-[10px]'
            } else if (BOOL_KEYS.includes(chip)) {
                def.valueFormatter = (params) => formatBoolean(params.value)
                def.cellClass = 'text-[var(--text-muted)] text-[10px]'
            } else if (METRIC_KEYS.includes(chip)) {
                def.cellRenderer = (params) => <MetricBadge value={params.value} />
                def.width = 80
            }

            defs.push(def)
        })

        return defs
    }, [activeChips])

    const defaultColDef = useMemo(() => ({
        sortable: true,
        filter: true,
        resizable: true,
        headerClass: 'text-[9px] uppercase tracking-widest text-[var(--text-muted)] font-black bg-[var(--bg-header)] opacity-80',
        cellClass: 'border-b border-[var(--border-main)] flex items-center py-2 transition-colors',
    }), [])

    // Synchronize selection when selectedTableName changes from outside
    useEffect(() => {
        if (gridRef.current?.api && selectedTableName) {
            gridRef.current.api.forEachNode(node => {
                if (node.data.table_name === selectedTableName) {
                    node.setSelected(true)
                }
            });
        }
    }, [selectedTableName])

    // ── Conditional renders come AFTER all hooks ──────────────────────────

    if (loading) {
        return (
            <div className="p-3 space-y-2 bg-[var(--bg-panel)]">
                {[...Array(6)].map((_, i) => (
                    <div key={i} className="h-10 rounded-lg bg-[var(--bg-card)] animate-pulse border border-[var(--border-main)] opacity-50" />
                ))}
            </div>
        )
    }

    if (error) {
        return (
            <div className="p-3 text-center bg-[var(--bg-panel)]">
                <p className="text-xs text-red-500 font-bold uppercase tracking-widest">Connection Interrupted</p>
                <p className="text-[10px] text-[var(--text-muted)] mt-1">{error}</p>
            </div>
        )
    }

    if (!rows || rows.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center py-8 bg-[var(--bg-panel)] gap-3 px-6 opacity-40">
                <div className="w-8 h-8 rounded-full border-2 border-[var(--text-muted)] border-dashed" />
                <p className="text-[10px] text-[var(--text-muted)] uppercase font-black tracking-[0.2em] text-center">
                    Awaiting Signal
                </p>
            </div>
        )
    }

    return (
        // domLayout="autoHeight" makes the grid size itself to fit content rows
        // instead of trying to fill 100% of the parent height (which is 0 when
        // the parent only has maxHeight and no explicit height set)
        <div className={cn(
            "w-full bg-[var(--bg-panel)] transition-colors",
            theme === 'dark' ? 'ag-theme-balham-dark' : 'ag-theme-balham'
        )}>
            <AgGridReact
                ref={gridRef}
                rowData={rows}
                columnDefs={columnDefs}
                defaultColDef={defaultColDef}
                rowSelection="single"
                onRowSelected={(event) => {
                    if (event.node.isSelected()) {
                        onSelect(event.node.data)
                    }
                }}
                animateRows={true}
                headerHeight={32}
                rowHeight={40}
                domLayout="autoHeight"
            />
        </div>
    )
}
