/**
 * Application header bar with title and a compact description.
 */
export default function Header() {
    return (
        <header className="border-b border-border bg-card px-6 py-3 flex items-center gap-3 shrink-0">
            <div className="flex items-center gap-2">
                {/* Minimal logo mark */}
                <div className="h-5 w-5 rounded bg-foreground/10 flex items-center justify-center">
                    <svg className="h-3 w-3 text-foreground/70" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <rect x="1" y="1" width="6" height="6" rx="1" />
                        <rect x="9" y="1" width="6" height="6" rx="1" />
                        <rect x="1" y="9" width="6" height="6" rx="1" />
                        <rect x="9" y="9" width="6" height="6" rx="1" />
                    </svg>
                </div>
                <span className="text-sm font-semibold tracking-tight text-foreground">RAG Metrics Explorer</span>
            </div>
            <span className="hidden sm:block text-xs text-muted-foreground">
                Browse, compare and export RAG experiment results
            </span>
        </header>
    )
}
