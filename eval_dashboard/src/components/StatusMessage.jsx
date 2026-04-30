/**
 * Full-page and inline status messages for loading / error / empty states.
 */
export default function StatusMessage({ loading, error, empty, emptyText = 'No data.' }) {
    if (loading) {
        return (
            <div className="flex items-center gap-2 py-12 justify-center text-muted-foreground text-sm">
                <span className="inline-block h-4 w-4 rounded-full border-2 border-current border-t-transparent animate-spin" />
                Loading…
            </div>
        )
    }
    if (error) {
        return (
            <div className="py-12 flex flex-col items-center gap-2 text-sm">
                <span className="text-destructive font-medium">Error</span>
                <span className="text-muted-foreground text-center max-w-xs">{error}</span>
            </div>
        )
    }
    if (empty) {
        return (
            <div className="py-12 text-center text-sm text-muted-foreground">
                {emptyText}
            </div>
        )
    }
    return null
}
