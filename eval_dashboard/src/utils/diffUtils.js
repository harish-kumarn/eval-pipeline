const METRIC_COLUMNS = [
    'mrr', 'recall', 'precision', 'ndcg',
    'faithfulness', 'context_precision', 'answer_correctness',
]

/**
 * Compute per-cell diffs between compare and base experiment rows.
 * Matches rows by `query` field.
 *
 * @param {Array} baseRows
 * @param {Array} compareRows
 * @param {string[]} metricColumns - columns to compute diff for
 * @returns {Map<string, Map<string, { delta: number, direction: 'up'|'down'|'same' }>>}
 */
export function computeDiff(baseRows, compareRows, metricColumns = METRIC_COLUMNS) {
    const result = new Map()

    if (!baseRows || !compareRows) return result

    // Build a lookup map from question → base row
    const baseByQuery = new Map()
    for (const row of baseRows) {
        if (row.query) baseByQuery.set(row.query, row)
    }

    for (const compareRow of compareRows) {
        const query = compareRow.query
        if (!query) continue

        const baseRow = baseByQuery.get(query)
        if (!baseRow) continue

        const cellDiffs = new Map()
        for (const col of metricColumns) {
            const baseVal = baseRow[col]
            const compareVal = compareRow[col]

            if (
                baseVal === null || baseVal === undefined || isNaN(baseVal) ||
                compareVal === null || compareVal === undefined || isNaN(compareVal)
            ) {
                cellDiffs.set(col, { delta: 0, direction: 'same' })
                continue
            }

            const delta = compareVal - baseVal
            let direction = 'same'
            if (delta > 0.00001) direction = 'up'
            else if (delta < -0.00001) direction = 'down'

            cellDiffs.set(col, { delta, direction })
        }

        result.set(query, cellDiffs)
    }

    return result
}

export { METRIC_COLUMNS }
