import { useState, useRef, useEffect, useMemo } from 'react'
import { createPortal } from 'react-dom'

const COLUMNS = [
    { key: 'timestamp', label: 'Timestamp' },
    { key: 'hybrid_search', label: 'Hybrid Search' },
    { key: 'query_reformulation', label: 'Query Reform.' },
    { key: 'reranking', label: 'Reranking' },
    { key: 'mrr_average', label: 'MRR Avg' },
    { key: 'recall_average', label: 'Recall Avg' },
    { key: 'precision_average', label: 'Precision Avg' },
    { key: 'ndcg_average', label: 'NDCG Avg' },
    { key: 'faithfulness_average', label: 'Faithful. Avg' },
    { key: 'context_precision_average', label: 'Ctx Prec. Avg' },
    { key: 'answer_correctness_average', label: 'Ans. Corr. Avg' },
]

/**
 * Filter bar with dropdowns for unique value selection.
 * Uses Portals to prevent clipping by overflow-x-auto.
 */
export default function ChipBar({ indexRows, columnFilters = {}, onFilterChange }) {
    return (
        <div className="relative mt-2">
            <div className="chip-fade-right flex gap-2 overflow-x-auto pb-1 flex-nowrap scrollbar-none">
                {COLUMNS.map((col) => (
                    <FilterDropdown
                        key={col.key}
                        col={col}
                        indexRows={indexRows}
                        selectedValues={columnFilters[col.key] || []}
                        onToggle={(val) => {
                            const current = columnFilters[col.key] || []
                            const next = current.includes(val)
                                ? current.filter((v) => v !== val)
                                : [...current, val]
                            onFilterChange(col.key, next)
                        }}
                    />
                ))}
            </div>
        </div>
    )
}

function FilterDropdown({ col, indexRows, selectedValues, onToggle }) {
    const [open, setOpen] = useState(false)
    const [coords, setCoords] = useState({ top: 0, left: 0 })
    const buttonRef = useRef(null)
    const dropdownRef = useRef(null)

    // Extract unique values
    const uniqueValues = useMemo(() => {
        if (!indexRows) return []
        const raw = indexRows.map(r => r[col.key])
        return Array.from(new Set(raw)).filter(v => v !== null && v !== undefined).sort((a, b) => {
            if (typeof a === 'number' && typeof b === 'number') return a - b
            return String(a).localeCompare(String(b))
        })
    }, [indexRows, col.key])

    // Handle open/position
    const toggleOpen = () => {
        if (!open && buttonRef.current) {
            const rect = buttonRef.current.getBoundingClientRect()
            setCoords({ top: rect.bottom + 4, left: rect.left })
        }
        setOpen(!open)
    }

    // Close on click outside or scroll
    useEffect(() => {
        if (!open) return
        function handler(e) {
            if (dropdownRef.current && !dropdownRef.current.contains(e.target) &&
                buttonRef.current && !buttonRef.current.contains(e.target)) {
                setOpen(false)
            }
        }
        // Also close on any scroll to avoid floating dropdowns
        function scrollHandler() { setOpen(false) }

        document.addEventListener('mousedown', handler)
        window.addEventListener('scroll', scrollHandler, true)
        return () => {
            document.removeEventListener('mousedown', handler)
            window.removeEventListener('scroll', scrollHandler, true)
        }
    }, [open])

    const active = selectedValues.length > 0

    return (
        <div className="relative flex-shrink-0">
            <button
                ref={buttonRef}
                onClick={toggleOpen}
                className={[
                    'px-2.5 py-1 rounded-lg text-xs font-bold transition-all duration-200 flex items-center gap-1.5',
                    'focus:outline-none focus:ring-2 focus:ring-[var(--accent)]/30',
                    active
                        ? 'bg-[var(--accent)] text-white shadow-lg shadow-amber-500/20 border border-[var(--accent)]'
                        : 'bg-transparent border border-[var(--border-main)] text-[var(--text-muted)] hover:border-[var(--text-main)] hover:text-[var(--text-main)]',
                ].join(' ')}
            >
                {col.label}
                {active && (
                    <span className="bg-white/20 dark:bg-black/20 px-1.5 rounded-md text-[9px] font-black min-w-[16px]">
                        {selectedValues.length}
                    </span>
                )}
                <svg className={`h-3 w-3 transition-transform ${open ? 'rotate-180' : ''}`} viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clipRule="evenodd" />
                </svg>
            </button>

            {open && createPortal(
                <div
                    ref={dropdownRef}
                    style={{ position: 'fixed', top: coords.top, left: coords.left }}
                    className="z-[100] w-48 bg-[var(--bg-panel)] border border-[var(--border-main)] rounded-xl shadow-2xl py-1.5 max-h-60 overflow-y-auto scrollbar-thin transition-colors"
                >
                    {uniqueValues.length === 0 ? (
                        <div className="px-3 py-2 text-xs text-gray-500">No values found</div>
                    ) : (
                        uniqueValues.map((val) => (
                            <label
                                key={val}
                                className="flex items-center gap-2 px-3 py-2 text-xs text-[var(--text-main)] hover:bg-gray-800/10 dark:hover:bg-white/5 cursor-pointer transition-colors"
                            >
                                <input
                                    type="checkbox"
                                    checked={selectedValues.includes(val)}
                                    onChange={() => onToggle(val)}
                                    className="accent-[var(--accent)] h-3.5 w-3.5 rounded-md focus:ring-0"
                                />
                                <span className="truncate font-mono">
                                    {typeof val === 'number' ? val.toFixed(4).replace(/\.?0+$/, '') : String(val)}
                                </span>
                            </label>
                        ))
                    )}
                </div>,
                document.body
            )}
        </div>
    )
}
