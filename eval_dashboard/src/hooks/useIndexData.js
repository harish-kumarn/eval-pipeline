import { useState, useEffect, useCallback } from 'react'
import { fetchIndex } from '../api/client'

/**
 * Fetches /api/index once on mount.
 * Returns: { data, loading, error, refetch }
 */
export function useIndexData() {
    const [data, setData] = useState([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    const load = useCallback(async () => {
        setLoading(true)
        setError(null)
        try {
            const rows = await fetchIndex()
            setData(rows)
        } catch (err) {
            setError(err.message || 'Failed to load index')
        } finally {
            setLoading(false)
        }
    }, [])

    useEffect(() => {
        load()
    }, [load])

    return { data, loading, error, refetch: load }
}
