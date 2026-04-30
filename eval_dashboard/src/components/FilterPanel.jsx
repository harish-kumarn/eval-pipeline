import { useState, useCallback, useMemo } from 'react'
import { cn } from '../utils/formatUtils'

// Filter type definitions per column category
const FILTER_TYPES = {
    text: [
        { value: 'contains', label: 'Contains' },
        { value: 'not_contains', label: 'Not Contains' },
        { value: 'equals', label: 'Equals' },
        { value: 'starts_with', label: 'Starts With' },
        { value: 'ends_with', label: 'Ends With' },
    ],
    number: [
        { value: 'equals', label: '= Equals' },
        { value: 'not_equals', label: '≠ Not Equals' },
        { value: 'gt', label: '> Greater Than' },
        { value: 'gte', label: '≥ Greater or Equal' },
        { value: 'lt', label: '< Less Than' },
        { value: 'lte', label: '≤ Less or Equal' },
        { value: 'between', label: '↔ Between' },
    ],
    boolean: [
        { value: 'true', label: '✅ Enabled' },
        { value: 'false', label: '❌ Disabled' },
    ],
}

const METRIC_COLS = ['mrr', 'recall', 'precision', 'ndcg', 'faithfulness', 'context_precision', 'answer_correctness']
const BOOL_COLS = ['hybrid_search', 'query_reformulation', 'reranking']

function getColCategory(col) {
    if (BOOL_COLS.includes(col)) return 'boolean'
    if (METRIC_COLS.includes(col) || col === 'top_k') return 'number'
    return 'text'
}

// ── Single filter row ────────────────────────────────────────────────────
function FilterRow({ columns, filter, onChange, onRemove }) {
    const category = getColCategory(filter.col)
    const filterTypes = FILTER_TYPES[category]
    const isBetween = filter.type === 'between'
    const isBoolean = category === 'boolean'

    return (
        <div className="flex items-center gap-2 flex-wrap">
            {/* Column selector */}
            <select
                value={filter.col}
                onChange={e => onChange({ ...filter, col: e.target.value, type: FILTER_TYPES[getColCategory(e.target.value)][0].value, value: '', value2: '' })}
                className="px-2 py-1.5 text-[11px] font-mono font-bold bg-[var(--bg-card)] border border-[var(--border-main)] rounded-lg text-[var(--text-main)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)] transition-colors"
            >
                {columns.map(c => (
                    <option key={c} value={c}>{c.toUpperCase().replace(/_/g, ' ')}</option>
                ))}
            </select>

            {/* Filter type selector */}
            {!isBoolean && (
                <select
                    value={filter.type}
                    onChange={e => onChange({ ...filter, type: e.target.value, value: '', value2: '' })}
                    className="px-2 py-1.5 text-[11px] font-bold bg-[var(--bg-card)] border border-[var(--border-main)] rounded-lg text-[var(--text-main)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)] transition-colors"
                >
                    {filterTypes.map(ft => (
                        <option key={ft.value} value={ft.value}>{ft.label}</option>
                    ))}
                </select>
            )}

            {/* Value inputs */}
            {!isBoolean && (
                <input
                    type={category === 'number' ? 'number' : 'text'}
                    step={category === 'number' ? '0.0001' : undefined}
                    placeholder={isBetween ? 'From…' : 'Value…'}
                    value={filter.value}
                    onChange={e => onChange({ ...filter, value: e.target.value })}
                    className="w-28 px-2 py-1.5 text-[11px] font-mono bg-[var(--bg-card)] border border-[var(--border-main)] rounded-lg text-[var(--text-main)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)] transition-colors placeholder-[var(--text-muted)]"
                />
            )}
            {isBetween && (
                <>
                    <span className="text-[10px] text-[var(--text-muted)] font-bold">and</span>
                    <input
                        type="number"
                        step="0.0001"
                        placeholder="To…"
                        value={filter.value2 || ''}
                        onChange={e => onChange({ ...filter, value2: e.target.value })}
                        className="w-24 px-2 py-1.5 text-[11px] font-mono bg-[var(--bg-card)] border border-[var(--border-main)] rounded-lg text-[var(--text-main)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)] transition-colors placeholder-[var(--text-muted)]"
                    />
                </>
            )}

            {/* Boolean options shown as radio-style pills */}
            {isBoolean && (
                <div className="flex gap-1.5">
                    {FILTER_TYPES.boolean.map(opt => (
                        <button
                            key={opt.value}
                            onClick={() => onChange({ ...filter, type: opt.value })}
                            className={cn(
                                'px-2.5 py-1 rounded-lg text-[11px] font-bold border transition-all',
                                filter.type === opt.value
                                    ? 'bg-[var(--accent)] text-white border-[var(--accent)]'
                                    : 'bg-[var(--bg-card)] text-[var(--text-muted)] border-[var(--border-main)] hover:border-[var(--accent)]'
                            )}
                        >
                            {opt.label}
                        </button>
                    ))}
                </div>
            )}

            {/* Remove button */}
            <button
                onClick={onRemove}
                className="ml-auto flex-shrink-0 p-1 text-[var(--text-muted)] hover:text-red-500 hover:bg-red-500/10 rounded-md transition-colors"
                title="Remove filter"
            >
                <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <path d="M18 6 6 18M6 6l12 12" />
                </svg>
            </button>
        </div>
    )
}

// ── Apply filter logic to a single row ──────────────────────────────────
function rowMatchesFilter(row, filter) {
    const raw = row[filter.col]
    const category = getColCategory(filter.col)

    if (category === 'boolean') {
        if (filter.type === 'true') return !!raw
        if (filter.type === 'false') return !raw
        return true
    }

    if (category === 'number') {
        if (filter.value === '' && filter.value !== 0) return true
        const v = Number(filter.value)
        const val = Number(raw)
        if (isNaN(val)) return false
        switch (filter.type) {
            case 'equals': return Math.abs(val - v) < 1e-9
            case 'not_equals': return Math.abs(val - v) >= 1e-9
            case 'gt': return val > v
            case 'gte': return val >= v
            case 'lt': return val < v
            case 'lte': return val <= v
            case 'between': {
                const v2 = Number(filter.value2 || Infinity)
                return val >= v && val <= v2
            }
            default: return true
        }
    }

    // text
    if (!filter.value) return true
    const cell = String(raw ?? '').toLowerCase()
    const query = filter.value.toLowerCase()
    switch (filter.type) {
        case 'contains': return cell.includes(query)
        case 'not_contains': return !cell.includes(query)
        case 'equals': return cell === query
        case 'starts_with': return cell.startsWith(query)
        case 'ends_with': return cell.endsWith(query)
        default: return true
    }
}

// ── Main FilterPanel component ───────────────────────────────────────────
let nextId = 1

export default function FilterPanel({ columns, data, onFilteredData }) {
    const [open, setOpen] = useState(false)
    const [filters, setFilters] = useState([])

    const addFilter = useCallback(() => {
        const col = columns[0] || 'query'
        const category = getColCategory(col)
        const type = FILTER_TYPES[category][0].value
        setFilters(prev => [...prev, { id: nextId++, col, type, value: '', value2: '' }])
        setOpen(true)
    }, [columns])

    const updateFilter = useCallback((id, updated) => {
        setFilters(prev => prev.map(f => f.id === id ? updated : f))
    }, [])

    const removeFilter = useCallback((id) => {
        setFilters(prev => prev.filter(f => f.id !== id))
    }, [])

    const clearAll = useCallback(() => {
        setFilters([])
    }, [])

    // Actually filter the data and notify parent
    const activeFilters = useMemo(() => {
        return filters.filter(f => {
            const cat = getColCategory(f.col)
            if (cat === 'boolean') return true
            return f.value !== ''
        })
    }, [filters])

    useMemo(() => {
        if (!data) { onFilteredData(null); return }
        if (activeFilters.length === 0) { onFilteredData(data); return }
        const filtered = data.filter(row => activeFilters.every(f => rowMatchesFilter(row, f)))
        onFilteredData(filtered)
    }, [data, activeFilters, onFilteredData])

    const activeCount = activeFilters.length

    return (
        <div className="shrink-0">
            {/* Trigger bar */}
            <div className="flex items-center gap-2 px-3 py-1.5 bg-[var(--bg-header)] border-b border-[var(--border-main)]">
                <button
                    onClick={() => setOpen(o => !o)}
                    className={cn(
                        'flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-bold border transition-all',
                        open
                            ? 'bg-[var(--accent)]/15 border-[var(--accent)]/40 text-[var(--accent)]'
                            : 'bg-[var(--bg-card)] border-[var(--border-main)] text-[var(--text-muted)] hover:text-[var(--text-main)] hover:border-[var(--text-muted)]'
                    )}
                >
                    <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M3 4h18M7 10h10M11 16h2" />
                    </svg>
                    Filters
                    {activeCount > 0 && (
                        <span className="px-1.5 py-0.5 rounded-md text-[9px] font-black bg-[var(--accent)] text-white">
                            {activeCount}
                        </span>
                    )}
                    <svg className={cn('w-3 h-3 transition-transform', open && 'rotate-180')} viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" />
                    </svg>
                </button>

                <button
                    onClick={addFilter}
                    className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-bold border border-dashed border-[var(--border-main)] text-[var(--text-muted)] hover:text-[var(--accent)] hover:border-[var(--accent)] transition-all bg-transparent"
                >
                    <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                        <path d="M12 5v14M5 12h14" />
                    </svg>
                    Add Filter
                </button>

                {filters.length > 0 && (
                    <button
                        onClick={clearAll}
                        className="px-2 py-1 rounded-md text-[11px] font-bold text-red-500/70 hover:text-red-500 hover:bg-red-500/10 transition-all"
                    >
                        Clear All
                    </button>
                )}

                {activeCount > 0 && (
                    <span className="ml-auto text-[10px] text-[var(--text-muted)] italic">
                        {activeCount} filter{activeCount > 1 ? 's' : ''} active
                    </span>
                )}
            </div>

            {/* Expanded filter rows */}
            {open && filters.length > 0 && (
                <div className="px-3 py-2.5 bg-[var(--bg-panel)] border-b border-[var(--border-main)] space-y-2">
                    {filters.map(f => (
                        <FilterRow
                            key={f.id}
                            columns={columns}
                            filter={f}
                            onChange={updated => updateFilter(f.id, { ...updated, id: f.id })}
                            onRemove={() => removeFilter(f.id)}
                        />
                    ))}
                </div>
            )}

            {open && filters.length === 0 && (
                <div className="px-4 py-3 bg-[var(--bg-panel)] border-b border-[var(--border-main)] text-center">
                    <p className="text-[10px] text-[var(--text-muted)] italic">
                        No filters yet — click <strong>Add Filter</strong> to start filtering rows.
                    </p>
                </div>
            )}
        </div>
    )
}
