import { useState } from 'react'

/**
 * CSV download button. Uses anchor-tag download as per spec (no window.open).
 * Props: { tableName: string | null }
 */
export default function ExportCSVButton({ tableName }) {
    const [clicked, setClicked] = useState(false)

    const handleClick = () => {
        if (!tableName) return
        const url = `${import.meta.env.VITE_API_BASE_URL}/api/experiment/${encodeURIComponent(tableName)}/csv`
        const a = document.createElement('a')
        a.href = url
        a.download = `${tableName}.csv`
        a.click()
        setClicked(true)
        setTimeout(() => setClicked(false), 1500)
    }

    const disabled = !tableName

    return (
        <button
            onClick={handleClick}
            disabled={disabled}
            className={[
                'inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-bold transition-all duration-200',
                'focus:outline-none focus:ring-2 focus:ring-[var(--accent)]/20 shadow-sm',
                disabled
                    ? 'bg-[var(--bg-card)] border border-[var(--border-main)] text-[var(--text-muted)] cursor-not-allowed opacity-50'
                    : 'bg-[var(--bg-app)] border border-[var(--border-main)] text-[var(--text-main)] hover:border-[var(--accent)] hover:text-[var(--accent)] cursor-pointer',
            ].join(' ')}
        >
            {clicked ? (
                <>
                    <span className="inline-block h-3 w-3 rounded-full border-2 border-current border-t-transparent animate-spin" />
                    Downloading…
                </>
            ) : (
                <>
                    <svg className="h-3.5 w-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M8 2v8M5 7l3 3 3-3" strokeLinecap="round" strokeLinejoin="round" />
                        <path d="M2 13h12" strokeLinecap="round" />
                    </svg>
                    ⬇ Export CSV
                </>
            )}
        </button>
    )
}
