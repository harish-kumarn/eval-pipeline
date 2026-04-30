import { useState } from 'react'
import { parsePillList } from '../utils/formatUtils'

/**
 * Renders a comma-separated list as inline pills.
 * Shows first 3 items; "+N more" pill with tooltip for the rest.
 * Props: { value: string | null }
 */
export default function PillList({ value }) {
    const [showTooltip, setShowTooltip] = useState(false)
    const items = parsePillList(value)

    if (!items.length) {
        return <span className="text-gray-500 font-mono text-xs">—</span>
    }

    const visible = items.slice(0, 3)
    const overflow = items.slice(3)

    return (
        <div className="flex flex-wrap gap-1 items-center">
            {visible.map((item, i) => (
                <span
                    key={i}
                    className="inline-block px-1.5 py-0.5 bg-[var(--bg-card)] text-[var(--text-main)] rounded-md text-[10px] font-mono border border-[var(--border-main)] transition-colors"
                >
                    {item}
                </span>
            ))}
            {overflow.length > 0 && (
                <div className="relative inline-block">
                    <span
                        className="inline-block px-1.5 py-0.5 bg-[var(--accent-muted)] text-[var(--accent)] rounded-md text-[10px] font-black font-mono cursor-default border border-[var(--accent)]/30 hover:bg-[var(--accent)] hover:text-white transition-all shadow-sm"
                        onMouseEnter={() => setShowTooltip(true)}
                        onMouseLeave={() => setShowTooltip(false)}
                    >
                        +{overflow.length} more
                    </span>
                    {showTooltip && (
                        <div className="absolute left-0 bottom-full mb-2 z-[60] min-w-max max-w-xs bg-[var(--bg-panel)] border border-[var(--border-main)] rounded-xl shadow-2xl p-2.5 animate-in fade-in slide-in-from-bottom-1 duration-200">
                            <p className="text-[9px] text-[var(--text-muted)] mb-1.5 font-black uppercase tracking-widest px-0.5">Hidden layers:</p>
                            <div className="flex flex-wrap gap-1.5">
                                {overflow.map((item, i) => (
                                    <span key={i} className="text-[10px] font-mono font-medium text-[var(--text-main)] bg-[var(--bg-card)] px-1.5 py-0.75 rounded-md border border-[var(--border-main)]">
                                        {item}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}
