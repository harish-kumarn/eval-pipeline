import { useState, useRef, useEffect } from 'react'
import { cn } from '../utils/formatUtils'

const FEATURE_CHIPS = [
    { key: 'hybrid_search', label: 'Hybrid Search' },
    { key: 'query_reformulation', label: 'Query Reform.' },
    { key: 'reranking', label: 'Reranking' },
]

/**
 * Combined filter bar: row search + feature flag chips + column toggle dropdown.
 * Props:
 *   search          – string
 *   setSearch       – fn(string)
 *   activeChips     – Set of active feature-flag keys
 *   toggleChip      – fn(key)
 *   allColumns      – string[] – all toggleable column keys
 *   hiddenColumns   – Set of hidden column keys
 *   toggleColumn    – fn(key)
 */
export default function FilterBar({
    search,
    setSearch,
    activeChips,
    toggleChip,
    allColumns,
    hiddenColumns,
    toggleColumn,
}) {
    const [colOpen, setColOpen] = useState(false)
    const dropRef = useRef(null)

    // Close on outside click
    useEffect(() => {
        function handler(e) {
            if (dropRef.current && !dropRef.current.contains(e.target)) setColOpen(false)
        }
        document.addEventListener('mousedown', handler)
        return () => document.removeEventListener('mousedown', handler)
    }, [])

    return (
        <div className="flex flex-wrap items-center gap-2 px-4 py-2 border-b border-border bg-card shrink-0">
            {/* Row search */}
            <input
                type="search"
                placeholder="Search questions…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className={[
                    'h-7 rounded border border-border bg-muted text-foreground',
                    'px-2.5 text-xs placeholder:text-muted-foreground w-52',
                    'focus:outline-none focus:ring-1 focus:ring-zinc-500 transition-colors',
                ].join(' ')}
            />

            {/* Feature-flag chips */}
            <div className="flex gap-1.5 flex-wrap">
                {FEATURE_CHIPS.map(({ key, label }) => (
                    <button
                        key={key}
                        onClick={() => toggleChip(key)}
                        className={cn(
                            'h-7 px-2.5 rounded border text-xs font-medium transition-colors duration-100 focus:outline-none focus:ring-1 focus:ring-zinc-500',
                            activeChips.has(key)
                                ? 'border-zinc-500 bg-zinc-700/40 text-zinc-200'
                                : 'border-border bg-muted text-muted-foreground hover:border-zinc-600 hover:text-foreground',
                        )}
                    >
                        {label}
                    </button>
                ))}
            </div>

            {/* Spacer */}
            <div className="flex-1 min-w-0" />

            {/* Column visibility dropdown */}
            <div className="relative" ref={dropRef}>
                <button
                    onClick={() => setColOpen((v) => !v)}
                    className={cn(
                        'h-7 px-2.5 rounded border border-border bg-muted text-xs text-muted-foreground',
                        'hover:bg-accent hover:text-foreground transition-colors duration-100',
                        'focus:outline-none focus:ring-1 focus:ring-zinc-500 flex items-center gap-1.5',
                    )}
                >
                    <svg className="h-3.5 w-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M1 4h14M1 8h14M1 12h14" strokeLinecap="round" />
                    </svg>
                    Columns
                    {hiddenColumns.size > 0 && (
                        <span className="ml-0.5 bg-zinc-600 text-zinc-200 rounded px-1 text-[10px]">
                            {allColumns.length - hiddenColumns.size}/{allColumns.length}
                        </span>
                    )}
                </button>

                {colOpen && (
                    <div className="absolute right-0 top-full mt-1 z-20 w-48 rounded border border-border bg-card shadow-lg shadow-black/40 py-1">
                        <p className="px-3 py-1.5 text-[10px] uppercase tracking-widest text-muted-foreground font-semibold border-b border-border mb-1">
                            Visible columns
                        </p>
                        {allColumns.map((col) => (
                            <label
                                key={col}
                                className="flex items-center gap-2 px-3 py-1.5 text-xs text-foreground/80 hover:bg-accent cursor-pointer"
                            >
                                <input
                                    type="checkbox"
                                    checked={!hiddenColumns.has(col)}
                                    onChange={() => toggleColumn(col)}
                                    className="accent-zinc-400 h-3 w-3"
                                />
                                <span className="font-mono">{col}</span>
                            </label>
                        ))}
                    </div>
                )}
            </div>
        </div>
    )
}
