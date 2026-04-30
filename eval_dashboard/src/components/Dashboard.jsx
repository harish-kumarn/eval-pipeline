import { useState, useCallback, useRef, useEffect } from 'react'
import { Link } from 'react-router-dom'
import Panel from './Panel'
import ThemeToggle from './ThemeToggle'

export default function Dashboard() {
    const [baseExperimentData, setBaseExperimentData] = useState(null)
    const [baseTableName, setBaseTableName] = useState(null)

    // ── Resizable splitter ────────────────────────────────────────────────
    const [leftPct, setLeftPct] = useState(50)
    const containerRef = useRef(null)
    const isDragging = useRef(false)

    const handleMouseDown = useCallback((e) => {
        e.preventDefault()
        isDragging.current = true
        document.body.style.cursor = 'col-resize'
        document.body.style.userSelect = 'none'
    }, [])

    useEffect(() => {
        const onMouseMove = (e) => {
            if (!isDragging.current || !containerRef.current) return
            const rect = containerRef.current.getBoundingClientRect()
            const pct = ((e.clientX - rect.left) / rect.width) * 100
            setLeftPct(Math.min(80, Math.max(20, pct)))
        }
        const onMouseUp = () => {
            if (isDragging.current) {
                isDragging.current = false
                document.body.style.cursor = ''
                document.body.style.userSelect = ''
            }
        }
        window.addEventListener('mousemove', onMouseMove)
        window.addEventListener('mouseup', onMouseUp)
        return () => {
            window.removeEventListener('mousemove', onMouseMove)
            window.removeEventListener('mouseup', onMouseUp)
        }
    }, [])

    const handleBaseLoad = useCallback((data) => setBaseExperimentData(data ?? null), [])
    const handleBaseTableName = useCallback((name) => setBaseTableName(name ?? null), [])

    return (
        // h-screen + flex-col: full viewport, no scroll on root
        <div className="flex flex-col h-screen bg-[var(--bg-app)] transition-colors duration-200 overflow-hidden">

            {/* ── App header ─────────────────────────────────────────────── */}
            <header className="shrink-0 flex items-center justify-between px-5 py-2.5 bg-[var(--bg-header)] border-b border-[var(--border-main)] shadow-sm z-50">
                <div className="flex items-center gap-3">
                    <svg className="h-4 w-4 text-[var(--accent)] shrink-0" viewBox="0 0 16 16" fill="currentColor">
                        <rect x="1" y="1" width="6" height="6" rx="0.5" opacity="0.7" />
                        <rect x="9" y="1" width="6" height="6" rx="0.5" />
                        <rect x="1" y="9" width="6" height="6" rx="0.5" />
                        <rect x="9" y="9" width="6" height="6" rx="0.5" opacity="0.7" />
                    </svg>
                    <span className="text-sm font-bold tracking-tight text-[var(--text-main)]">RAG Metrics Explorer</span>
                    <span className="text-[11px] text-[var(--text-muted)] hidden md:block">
                        Advanced RAG evaluation and regression testing dashboard
                    </span>
                </div>
                <div className="flex items-center gap-4">
                    <ThemeToggle />
                    <Link
                        to="/analytics"
                        className="flex items-center gap-2 px-3 py-1.5 bg-indigo-600/10 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800/50 rounded-lg text-[11px] font-bold text-indigo-700 dark:text-indigo-400 hover:bg-indigo-600/20 transition-all uppercase tracking-wider"
                    >
                        <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                            <path d="M3 3v18h18M7 16l4-4 4 4 5-5" />
                        </svg>
                        Analytics
                    </Link>
                </div>
            </header>

            {/* ── Dual-panel layout ─────────────────────────────────────────
                flex-1 + min-h-0 = fills remaining height, allows children to shrink
            ───────────────────────────────────────────────────────────────── */}
            <div ref={containerRef} className="flex flex-1 min-h-0 overflow-hidden">

                {/* Base panel — width driven by leftPct state */}
                <div
                    style={{ width: `${leftPct}%` }}
                    className="flex flex-col min-h-0 min-w-0 overflow-hidden"
                >
                    <Panel
                        side="base"
                        onExperimentLoad={handleBaseLoad}
                        onTableNameChange={handleBaseTableName}
                    />
                </div>

                {/* ── Resizable divider ──────────────────────────────────── */}
                <div
                    onMouseDown={handleMouseDown}
                    className="group relative w-2 shrink-0 bg-[var(--border-main)] hover:bg-[var(--accent)] transition-colors duration-150 cursor-col-resize z-20 select-none"
                    title="Drag to resize panels"
                >
                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                        {[0, 1, 2, 3, 4].map(i => (
                            <div key={i} className="w-0.5 h-0.5 rounded-full bg-white/60" />
                        ))}
                    </div>
                    {/* Wider invisible hit area */}
                    <div className="absolute inset-y-0 -left-2 -right-2" />
                </div>

                {/* Compare panel */}
                <div
                    style={{ width: `${100 - leftPct}%` }}
                    className="flex flex-col min-h-0 min-w-0 overflow-hidden"
                >
                    <Panel
                        side="compare"
                        baseData={baseExperimentData}
                        baseTableName={baseTableName}
                        onExperimentLoad={() => { }}
                    />
                </div>
            </div>
        </div>
    )
}
