import { useState, useRef, useEffect } from 'react'
import { createPortal } from 'react-dom'

const COLUMN_GROUPS = [
    {
        label: 'Core',
        cols: ['id', 'timestamp', 'query', 'type', 'top_k', 'reformulated_query'],
    },
    {
        label: 'Retrieval',
        cols: [
            'vector_chunk_ids', 'bm25_chunk_ids',
            'pre_rerank_chunk_ids', 'post_rerank_chunk_ids',
            'final_retrieved_chunks', 'actual_chunk',
        ],
    },
    {
        label: 'Metrics',
        cols: [
            'mrr', 'recall', 'precision', 'ndcg',
            'faithfulness', 'context_precision', 'answer_correctness',
        ],
    },
]

/**
 * Column visibility popover with grouped checkboxes.
 * Uses Portals to prevent clipping by parent overflow-hidden.
 * Props: { allColumns: string[], visibleColumns: string[], onToggle: fn(col) }
 */
export default function ColumnDropdown({ allColumns, visibleColumns, onToggle }) {
    const [open, setOpen] = useState(false)
    const [coords, setCoords] = useState({ top: 0, right: 0, maxHSec: '70vh' })
    const buttonRef = useRef(null)
    const dropdownRef = useRef(null)

    // Calculate position and available height when opening
    const toggleOpen = () => {
        if (!open && buttonRef.current) {
            const rect = buttonRef.current.getBoundingClientRect()
            const availableBottom = window.innerHeight - rect.bottom - 20
            setCoords({
                top: rect.bottom + 4,
                right: window.innerWidth - rect.right,
                maxHSec: `${Math.max(100, availableBottom)}px`,
            })
        }
        setOpen(!open)
    }

    useEffect(() => {
        if (!open) return
        function handler(e) {
            if (dropdownRef.current && !dropdownRef.current.contains(e.target) &&
                buttonRef.current && !buttonRef.current.contains(e.target)) {
                setOpen(false)
            }
        }

        function scrollHandler(e) {
            // Only close if scrolling outside the dropdown
            if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
                setOpen(false)
            }
        }

        document.addEventListener('mousedown', handler)
        window.addEventListener('scroll', scrollHandler, true)
        return () => {
            document.removeEventListener('mousedown', handler)
            window.removeEventListener('scroll', scrollHandler, true)
        }
    }, [open])

    const visibleCount = visibleColumns.length

    return (
        <div className="relative">
            <button
                ref={buttonRef}
                onClick={toggleOpen}
                className="inline-flex items-center gap-1.5 px-2.5 py-1.5 bg-[var(--bg-app)] border border-[var(--border-main)] rounded-lg
                   text-xs text-[var(--text-main)] hover:border-[var(--accent)] hover:text-[var(--accent)] transition-all
                   focus:outline-none focus:ring-2 focus:ring-[var(--accent)]/20 shadow-sm"
            >
                <svg className="h-3.5 w-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <rect x="1" y="3" width="14" height="1.5" rx="0.5" />
                    <rect x="1" y="7.25" width="14" height="1.5" rx="0.5" />
                    <rect x="1" y="11.5" width="14" height="1.5" rx="0.5" />
                </svg>
                Columns ▾
                <span className="ml-1 text-[10px] bg-[var(--bg-panel)] text-[var(--text-muted)] rounded-md px-1.5 border border-[var(--border-main)]">
                    {visibleCount}/{allColumns.length}
                </span>
            </button>

            {open && createPortal(
                <div
                    ref={dropdownRef}
                    style={{
                        position: 'fixed',
                        top: coords.top,
                        right: coords.right,
                        maxHeight: coords.maxHSec,
                        overflowY: 'scroll', // Forced scroll as requested
                    }}
                    className="z-[100] w-52 bg-[var(--bg-panel)] border border-[var(--border-main)]
                        rounded-xl shadow-2xl shadow-black/80 py-1 scrollbar-thin transition-colors"
                >
                    <div className="px-3 py-2 border-b border-[var(--border-main)] flex items-center justify-between bg-[var(--bg-header)] sticky top-0 z-20">
                        <span className="text-[10px] font-black text-[var(--text-muted)] uppercase tracking-tight">Visibility</span>
                        <button
                            onClick={(e) => {
                                e.stopPropagation();
                                const allInGroups = COLUMN_GROUPS.flatMap(g => g.cols);
                                const someMissing = allInGroups.some(c => !visibleColumns.includes(c));
                                if (someMissing) {
                                    allInGroups.forEach(c => {
                                        if (!visibleColumns.includes(c)) onToggle(c);
                                    });
                                } else {
                                    allInGroups.forEach(c => {
                                        if (visibleColumns.includes(c)) onToggle(c);
                                    });
                                }
                            }}
                            className="text-[10px] text-[var(--accent)] hover:underline font-black uppercase transition-colors"
                        >
                            {COLUMN_GROUPS.flatMap(g => g.cols).every(c => visibleColumns.includes(c)) ? 'None' : 'All'}
                        </button>
                    </div>
                    {COLUMN_GROUPS.map((group) => {
                        const groupCols = group.cols.filter((c) => allColumns.includes(c))
                        if (!groupCols.length) return null
                        return (
                            <div key={group.label} className="border-b border-[var(--border-main)] last:border-b-0 pb-1 mb-1 last:mb-0 last:pb-0">
                                <p className="px-3 pt-3 pb-1.5 text-[10px] uppercase tracking-widest text-[var(--text-muted)] font-black bg-[var(--bg-header)]/50">
                                    {group.label}
                                </p>
                                {groupCols.map((col) => {
                                    const isVisible = visibleColumns.includes(col);
                                    return (
                                        <label
                                            key={col}
                                            className={`flex items-center gap-3 px-3 py-2.5 text-xs cursor-pointer transition-all
                                                ${isVisible ? 'bg-[var(--accent-muted)] text-[var(--text-main)]' : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'}
                                                hover:bg-gray-800/10 dark:hover:bg-white/5 group`}
                                        >
                                            <div className={`shrink-0 w-4 h-4 rounded-md border flex items-center justify-center transition-all
                                                ${isVisible ? 'bg-[var(--accent)] border-[var(--accent)] shadow-sm' : 'border-[var(--border-main)] bg-[var(--bg-app)] group-hover:border-[var(--text-muted)]'}`}>
                                                {isVisible && (
                                                    <svg className="w-3 h-3 text-white" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="3">
                                                        <polyline points="2 6 5 9 10 3" />
                                                    </svg>
                                                )}
                                            </div>
                                            <input
                                                type="checkbox"
                                                checked={isVisible}
                                                onChange={() => onToggle(col)}
                                                className="hidden"
                                            />
                                            <span className="font-semibold truncate">{col}</span>
                                        </label>
                                    );
                                })}
                            </div>
                        )
                    })}
                </div>,
                document.body
            )}
        </div>
    )
}
