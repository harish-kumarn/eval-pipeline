const BASE = import.meta.env.VITE_API_BASE_URL

class ApiError extends Error {
    constructor(message, status) {
        super(message)
        this.name = 'ApiError'
        this.status = status
    }
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 30000) {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs)
    try {
        const response = await fetch(url, { ...options, signal: controller.signal })
        if (!response.ok) {
            let errMsg = `HTTP ${response.status}`
            try {
                const body = await response.json()
                errMsg = body.detail || body.message || errMsg
            } catch (_) { }
            throw new ApiError(errMsg, response.status)
        }
        return response
    } finally {
        clearTimeout(timeoutId)
    }
}

/**
 * GET /api/index
 * Returns all rows from the index table
 */
export async function fetchIndex() {
    const res = await fetchWithTimeout(`${BASE}/api/index`)
    return res.json()
}

/**
 * GET /api/experiment/:tableName
 * Returns all rows from the named experiment table
 */
export async function fetchExperiment(tableName) {
    if (!tableName) throw new ApiError('No table name provided', 400)
    const res = await fetchWithTimeout(`${BASE}/api/experiment/${encodeURIComponent(tableName)}`)
    return res.json()
}

/**
 * Returns the CSV download URL for an experiment table
 */
export function getExperimentCSVUrl(tableName) {
    return `${BASE}/api/experiment/${encodeURIComponent(tableName)}/csv`
}
