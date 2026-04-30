/**
 * Search bar for filtering experiment list by table_name.
 * Props: { value: string, onChange: fn(string) }
 */
export default function SearchBar({ value, onChange }) {
    return (
        <div className="relative">
            <svg
                className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-gray-500 pointer-events-none"
                viewBox="0 0 16 16"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
            >
                <circle cx="6.5" cy="6.5" r="4.5" />
                <path d="M10.5 10.5l3 3" strokeLinecap="round" />
            </svg>
            <input
                type="search"
                placeholder="Search experiments…"
                value={value}
                onChange={(e) => onChange(e.target.value)}
                className="w-full bg-[var(--bg-app)] border border-[var(--border-main)] rounded-lg text-[var(--text-main)] placeholder-[var(--text-muted)]
                   text-xs pl-8 pr-3 py-2 focus:outline-none focus:border-[var(--accent)]
                   focus:ring-2 focus:ring-[var(--accent)]/20 transition-all shadow-sm"
            />
        </div>
    )
}
