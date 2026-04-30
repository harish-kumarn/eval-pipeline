import { useState, useEffect } from 'react'
import { fetchExperiment } from '../api/client'

/**
 * Fetches /api/experiment/:tableName on demand.
 * Re-fetches whenever tableName changes.
 * tableName = null means no fetch.
 * Returns: { data, loading, error }
 */
export function useExperiment(tableName) {
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    useEffect(() => {
        if (!tableName) {
            setData(null)
            setLoading(false)
            setError(null)
            return
        }

        let cancelled = false
        setLoading(true)
        setError(null)
        setData(null)

        fetchExperiment(tableName)
            .then((rows) => {
                if (!cancelled) setData(rows)
            })
            .catch((err) => {
                if (!cancelled) setError(err.message || 'Failed to load experiment')
            })
            .finally(() => {
                if (!cancelled) setLoading(false)
            })

        return () => { cancelled = true }
    }, [tableName])

    return { data, tableName, loading, error }
}
