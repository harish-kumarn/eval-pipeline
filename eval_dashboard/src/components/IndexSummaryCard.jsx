import { formatMetric, formatBoolean, formatTimestamp } from '../utils/formatUtils'
import MetricBadge from './MetricBadge'

const BOOL_COLS = ['hybrid_search', 'query_reformulation', 'reranking']
const METRIC_COLS = [
    'mrr_average', 'recall_average', 'precision_average', 'ndcg_average',
    'faithfulness_average', 'context_precision_average', 'answer_correctness_average',
]

/**
 * Summary card for the selected index row.
 * Props: { row: IndexRow | null, side: "base"|"compare", baseTableName?: string }
 */
export default function IndexSummaryCard({ row, side, baseTableName }) {
    const isCompare = side === 'compare'
    const accentClass = isCompare ? 'text-cyan-400 border-cyan-800' : 'text-amber-400 border-amber-800'
    const labelClass = isCompare ? 'text-cyan-500' : 'text-amber-500'

    if (!row) {
        return (
            <div className="px-4 py-6 border border-dashed border-[var(--border-main)] rounded-xl mx-3 my-2 bg-[var(--bg-panel)] opacity-50 flex flex-col items-center justify-center gap-2">
                <div className="p-2 bg-[var(--bg-app)] rounded-full">
                    <svg className="w-5 h-5 text-[var(--accent)] opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                </div>
                <p className="text-[10px] font-black text-[var(--text-muted)] text-center uppercase tracking-[0.2em]">
                    Intelligence Pending
                </p>
                <p className="text-[9px] text-[var(--text-muted)] italic text-center leading-tight">
                    Select an experiment above<br />to reveal data metrics
                </p>
            </div>
        )
    }

    return (
        <div className={`mx-3 my-2 border rounded-xl bg-[var(--bg-panel)] overflow-hidden border-[var(--border-main)] shadow-sm ring-1 ring-black/5 dark:ring-white/5 transition-all`}>
            {/* Header */}
            <div className={`px-3 py-1.5 border-b border-[var(--border-main)] flex items-center justify-between bg-[var(--bg-header)] transition-colors`}>
                <div className="flex items-center gap-2">
                    <div className={`w-1.5 h-1.5 rounded-full ${isCompare ? 'bg-cyan-500' : 'bg-amber-500'}`} />
                    <span className={`text-[10px] font-black uppercase tracking-widest text-[var(--text-main)]`}>
                        {row.table_name}
                    </span>
                </div>
                {isCompare && baseTableName && (
                    <span className="text-[9px] font-black bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border border-cyan-500/20 rounded-md px-2 py-0.5 uppercase tracking-tighter">
                        vs {baseTableName}
                    </span>
                )}
            </div>

            {/* Metadata grid */}
            <div className="px-3.5 py-4 grid grid-cols-2 gap-x-8 gap-y-3">
                {/* Timestamp */}
                <SummaryRow label="Time" value={formatTimestamp(row.timestamp)} />

                {/* Boolean flags */}
                {BOOL_COLS.map((col) => (
                    <SummaryRow
                        key={col}
                        label={col.replace(/_/g, ' ')}
                        value={formatBoolean(row[col])}
                        isStatus
                        statusValue={row[col]}
                    />
                ))}

                {/* Metric averages */}
                <div className="col-span-2 grid grid-cols-2 gap-x-8 gap-y-2 mt-2 pt-4 border-t border-[var(--border-main)] border-dashed">
                    {METRIC_COLS.map((col) => (
                        <div key={col} className="flex items-center justify-between gap-2">
                            <span className="text-[9px] font-black text-[var(--text-muted)] truncate uppercase tracking-tight opacity-70">
                                {col.replace(/_average$/, '').replace(/_/g, ' ')}
                            </span>
                            <MetricBadge value={row[col]} />
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}

function SummaryRow({ label, value, isStatus, statusValue }) {
    return (
        <div className="flex flex-col gap-1">
            <span className="text-[9px] font-black text-[var(--text-muted)] truncate uppercase tracking-widest opacity-60">{label}</span>
            <div className="flex items-center gap-1.5">
                {isStatus && (
                    <div className={`w-1 h-1 rounded-full ${statusValue ? 'bg-green-500 shadow-[0_0_5px_rgba(34,197,94,0.5)]' : 'bg-red-500 opacity-50'}`} />
                )}
                <span className={`text-[10px] text-[var(--text-main)] font-mono font-black ${isStatus && !statusValue ? 'opacity-40' : ''}`}>{value}</span>
            </div>
        </div>
    )
}
