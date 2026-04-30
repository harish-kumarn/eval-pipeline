import { useState } from 'react'
import StatusMessage from './StatusMessage'
import ExperimentCard from './ExperimentCard'

/**
 * Sidebar listing all experiments from the index.
 * Props:
 *   experiments   – array of index rows
 *   loading       – bool
 *   error         – string | null
 *   selected      – currently selected row (or null)
 *   compareRow    – currently compared row (or null)
 *   onSelect      – fn(row)
 *   onCompare     – fn(row)
 */
export default function Sidebar({
    experiments,
    loading,
    error,
    selected,
    compareRow,
    onSelect,
    onCompare,
}) {
    const [search, setSearch] = useState('')

    const filtered = experiments.filter((e) =>
        e.table_name.toLowerCase().includes(search.toLowerCase()),
    )

    return (
        <aside className="flex flex-col w-64 shrink-0 border-r border-border bg-card h-full overflow-hidden">
            {/* Sidebar header */}
            <div className="px-4 py-3 border-b border-border shrink-0">
                <p className="text-[11px] font-semibold uppercase tracking-widest text-muted-foreground mb-2">
                    Experiments
                </p>
                <input
                    type="search"
                    placeholder="Filter…"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    className={[
                        'w-full rounded border border-border bg-muted text-foreground',
                        'px-2.5 py-1.5 text-xs placeholder:text-muted-foreground',
                        'focus:outline-none focus:ring-1 focus:ring-zinc-500 transition-colors',
                    ].join(' ')}
                />
            </div>

            {/* Experiment list */}
            <div className="flex-1 overflow-y-auto">
                {loading || error || filtered.length === 0 ? (
                    <StatusMessage
                        loading={loading}
                        error={error}
                        empty={!loading && !error && filtered.length === 0}
                        emptyText={search ? 'No experiments match.' : 'No experiments found.'}
                    />
                ) : (
                    filtered.map((row) => (
                        <ExperimentCard
                            key={row.table_name}
                            row={row}
                            selected={selected?.table_name === row.table_name}
                            isCompare={compareRow?.table_name === row.table_name}
                            onSelect={onSelect}
                            onCompare={onCompare}
                        />
                    ))
                )}
            </div>

            {/* Count footer */}
            {!loading && !error && (
                <div className="px-4 py-2 border-t border-border shrink-0">
                    <span className="text-[10px] text-muted-foreground">
                        {filtered.length} of {experiments.length} experiment{experiments.length !== 1 ? 's' : ''}
                    </span>
                </div>
            )}
        </aside>
    )
}
