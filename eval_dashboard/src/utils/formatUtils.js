/**
 * Format a numeric metric value to 4 decimal places, or "—" for null/NaN.
 */
export function formatMetric(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
    return Number(value).toFixed(4)
}

/**
 * Format a boolean 0/1 value as "1" or "0".
 */
export function formatBoolean(value) {
    return value ? '1' : '0'
}

/**
 * Parse DD_MM_HH_MM timestamp string to human-readable format.
 * Input: "01_06_14_32"  →  Output: "01/06 14:32"
 */
export function formatTimestamp(value) {
    if (!value) return '—'
    const parts = value.split('_')
    if (parts.length !== 4) return value
    const [dd, mm, hh, min] = parts
    return `${dd}/${mm} ${hh}:${min}`
}

/**
 * Parse comma-separated TEXT field or JSON-style array string into array of trimmed strings.
 */
export function parsePillList(value) {
    if (!value) return []
    // If it's a bracketed list string like "[1, 2, 3]", remove brackets
    let cleaned = String(value).trim()
    if (cleaned.startsWith('[') && cleaned.endsWith(']')) {
        cleaned = cleaned.substring(1, cleaned.length - 1)
    }
    return cleaned.split(',').map((s) => s.trim()).filter(Boolean)
}

/**
 * Returns color category based on metric value thresholds.
 * @returns {"green"|"yellow"|"red"|"grey"}
 */
export function getMetricColor(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return 'grey'
    const n = Number(value)
    if (n >= 0.8) return 'green'
    if (n >= 0.5) return 'yellow'
    return 'red'
}

/**
 * Lightweight utility to compose class strings.
 */
export function cn(...classes) {
    return classes.filter(Boolean).join(' ')
}
