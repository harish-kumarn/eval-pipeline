import { useMemo } from 'react'
import { formatMetric, formatTimestamp, getMetricColorClass } from '../utils/formatUtils'
import MetricBadge from './MetricBadge'
import { cn } from '../utils/formatUtils'

const PREVIEW_METRICS = ['mrr', 'recall', 'ndcg', 'answer_correctness']

/**
 * A single experiment card in the sidebar.
 * Props:
 *   row            – index table row object
 *   selected       – bool, this experiment is the primary selection
 *   isCompare      – bool, this experiment is the comparison target
 *   onSelect       – fn(row) – set as primary
 *   onCompare      – fn(row) – set as compare / unset if already compare
 */
export default function ExperimentCard({ row, selected, isCompare, onSelect, onCompare }) {
    const badges = useMemo(() =>
        PREVIEW_METRICS.map((key) => {
            const val = row[`${key}_average`]
            return { key, val, colorClass: getMetricColorClass(val) }
        }), [row])

    const handleCompare = (e) => {
        e.stopPropagation()
        onCompare(row)
    }

    const timestamp = formatTimestamp(row.timestamp)

    return (
        <button
            onClick={() => onSelect(row)}
            aria-pressed={selected}
            className={cn(
                'w-full text-left px-4 py-3 border-b border-[var(--border-main)] transition-colors duration-100',
                'focus:outline-none focus:ring-1 focus:ring-inset focus:ring-[var(--accent)]/40',
                selected
                    ? 'bg-[var(--accent-muted)] text-[var(--text-main)]'
                    : 'bg-[var(--bg-card)] text-[var(--text-main)]/80 hover:bg-[var(--accent-muted)]/50',
            )}
        >
            {/* Name + timestamp */}
            <div className="flex items-start justify-between gap-2 mb-1.5">
                <span className="text-xs font-bold font-mono truncate max-w-[160px] text-[var(--text-main)]">
                    {row.table_name}
                </span>
                <span className="text-[10px] text-[var(--text-muted)] shrink-0">{timestamp}</span>
            </div>

            {/* Feature flags */}
            <div className="flex gap-3 flex-wrap mb-2 text-[9px] font-mono font-black uppercase tracking-widest opacity-80">
                <div className="flex items-center gap-1">
                    <span className="text-[var(--text-muted)]">HYB:</span>
                    <span className={row.hybrid_search === 1 ? "text-[var(--accent)] font-black" : "text-[var(--text-muted)]/40"}>{row.hybrid_search}</span>
                </div>
                <div className="flex items-center gap-1">
                    <span className="text-[var(--text-muted)]">REF:</span>
                    <span className={row.query_reformulation === 1 ? "text-[var(--accent)] font-black" : "text-[var(--text-muted)]/40"}>{row.query_reformulation}</span>
                </div>
                <div className="flex items-center gap-1">
                    <span className="text-[var(--text-muted)]">RRK:</span>
                    <span className={row.reranking === 1 ? "text-[var(--accent)] font-black" : "text-[var(--text-muted)]/40"}>{row.reranking}</span>
                </div>
            </div>

            {/* Metric previews */}
            <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
                {badges.map(({ key, val, colorClass }) => (
                    <MetricBadge
                        key={key}
                        label={key.toUpperCase()}
                        value={formatMetric(val)}
                        colorClass={colorClass}
                    />
                ))}
            </div>

            {/* Compare button */}
            <div className="mt-2 flex justify-end">
                <span
                    role="button"
                    tabIndex={0}
                    onClick={handleCompare}
                    onKeyDown={(e) => e.key === 'Enter' && handleCompare(e)}
                    className={cn(
                        'text-[10px] px-2 py-0.5 rounded border transition-colors duration-100 font-bold uppercase tracking-tighter',
                        isCompare
                            ? 'border-[var(--accent)] text-[var(--accent)] bg-[var(--accent-muted)]'
                            : 'border-[var(--border-main)] text-[var(--text-muted)] hover:border-[var(--accent)] hover:text-[var(--accent)]',
                    )}
                >
                    {isCompare ? '✓ Comparing' : 'Compare'}
                </span>
            </div>
        </button>
    )
}

function Chip({ label }) {
    return (
        <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-muted text-muted-foreground border border-border">
            {label}
        </span>
    )
}
