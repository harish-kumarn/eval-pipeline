import { formatMetric, getMetricColor } from '../utils/formatUtils'

const COLOR_MAP = {
    green: 'bg-[var(--metric-green-bg)] text-[var(--metric-green-text)] border-[var(--metric-green-border)]',
    yellow: 'bg-[var(--metric-yellow-bg)] text-[var(--metric-yellow-text)] border-[var(--metric-yellow-border)]',
    red: 'bg-[var(--metric-red-bg)] text-[var(--metric-red-text)] border-[var(--metric-red-border)]',
    grey: 'bg-[var(--metric-grey-bg)] text-[var(--metric-grey-text)] border-[var(--metric-grey-border)]',
}

/**
 * Colored metric value badge.
 * Props: { label?: string, value: number | string | null, colorClass?: string }
 */
export default function MetricBadge({ label, value, colorClass }) {
    const color = getMetricColor(value)
    const activeColorClass = colorClass || COLOR_MAP[color]

    return (
        <div className="flex items-center gap-1.5 min-w-0">
            {label && (
                <span className="text-[9px] font-black uppercase text-[var(--text-muted)] opacity-60 tracking-tighter shrink-0 truncate">
                    {label}:
                </span>
            )}
            <span
                className={`inline-block px-1.5 py-0.5 rounded-md text-[10px] font-black font-mono tabular-nums border transition-colors ${activeColorClass}`}
            >
                {typeof value === 'number' ? formatMetric(value) : value}
            </span>
        </div>
    )
}
