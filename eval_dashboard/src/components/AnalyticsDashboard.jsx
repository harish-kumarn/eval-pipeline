import React, { useState, useEffect, useMemo } from 'react'
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    ScatterChart, Scatter, ZAxis, Legend, Cell, AreaChart, Area
} from 'recharts'
import { LayoutDashboard, TrendingUp, Target, Activity, ArrowLeft, Loader2, AlertCircle } from 'lucide-react'
import { Link } from 'react-router-dom'
import { useTheme } from '../context/ThemeContext'
import ThemeToggle from './ThemeToggle'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001'

const METRIC_COLORS = {
    mrr: '#fbbf24', // amber-400
    recall: '#22d3ee', // cyan-400
    precision: '#818cf8', // indigo-400
    ndcg: '#34d399', // emerald-400
    faithfulness: '#f472b6', // pink-400
    context_precision: '#a78bfa', // violet-400
    answer_correctness: '#fb7185' // rose-400
}

const METRIC_LABELS = {
    mrr: 'MRR',
    recall: 'Recall',
    precision: 'Precision',
    ndcg: 'NDCG',
    faithfulness: 'Faithfulness',
    context_precision: 'Context Precision',
    answer_correctness: 'Answer Correctness'
}

export default function AnalyticsDashboard() {
    const [trends, setTrends] = useState([])
    const [topkData, setTopkData] = useState([])
    const [scatterData, setScatterData] = useState([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const { theme } = useTheme()

    useEffect(() => {
        const fetchAnalytics = async () => {
            try {
                const [trendsRes, topkRes, scatterRes] = await Promise.all([
                    fetch(`${API_BASE_URL}/api/analytics/trends`),
                    fetch(`${API_BASE_URL}/api/analytics/topk`),
                    fetch(`${API_BASE_URL}/api/analytics/scatter`)
                ])

                if (!trendsRes.ok || !topkRes.ok || !scatterRes.ok) throw new Error('Failed to fetch analytics data')

                const [trendsData, tkData, scData] = await Promise.all([
                    trendsRes.json(),
                    topkRes.json(),
                    scatterRes.json()
                ])

                setTrends(trendsData)
                setTopkData(tkData)
                setScatterData(scData)
            } catch (err) {
                setError(err.message)
            } finally {
                setLoading(false)
            }
        }
        fetchAnalytics()
    }, [])

    const globalStats = useMemo(() => {
        if (trends.length === 0) return {}
        const latest = trends[trends.length - 1]
        const averages = {}
        Object.keys(METRIC_COLORS).forEach(m => {
            const sum = trends.reduce((acc, curr) => acc + (curr[`${m}_average`] || 0), 0)
            averages[m] = {
                current: latest[`${m}_average`],
                avg: sum / trends.length,
                trend: trends.map(t => ({ value: t[`${m}_average`] }))
            }
        })
        return averages
    }, [trends])

    if (loading) return (
        <div className="flex flex-col items-center justify-center h-full bg-[var(--bg-app)] text-[var(--text-muted)]">
            <Loader2 className="h-8 w-8 animate-spin mb-4 text-[var(--accent)]" />
            <p className="text-sm font-medium animate-pulse">Computing Intelligence Layers...</p>
        </div>
    )

    if (error) return (
        <div className="flex flex-col items-center justify-center h-full bg-[var(--bg-app)] text-red-500 p-6 text-center">
            <AlertCircle className="h-12 w-12 mb-4 opacity-50" />
            <h3 className="text-lg font-bold text-[var(--text-main)] mb-2">Analytics Core Offline</h3>
            <p className="text-sm opacity-80 max-w-md mb-6">{error}</p>
            <Link to="/" className="px-4 py-2 bg-[var(--bg-panel)] border border-[var(--border-main)] rounded-lg hover:border-[var(--accent)] transition-all flex items-center gap-2">
                <ArrowLeft className="h-4 w-4" /> Return to Safety
            </Link>
        </div>
    )

    return (
        <div className="flex flex-col h-full bg-[var(--bg-app)] text-[var(--text-main)] overflow-y-auto scrollbar-thin transition-colors">
            {/* Header */}
            <header className="shrink-0 flex items-center justify-between px-6 py-4 bg-[var(--bg-header)] border-b border-[var(--border-main)] sticky top-0 z-20 shadow-sm transition-colors">
                <div className="flex items-center gap-4">
                    <Link to="/" className="p-2 hover:bg-gray-800/10 dark:hover:bg-white/5 rounded-full transition-colors text-[var(--text-muted)] hover:text-[var(--text-main)]">
                        <ArrowLeft className="h-5 w-5" />
                    </Link>
                    <div>
                        <h1 className="text-xl font-extrabold tracking-tight">System Intelligence</h1>
                        <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-widest font-bold mt-0.5">Aggregate Performance & Cross-Experiment Sensitivity</p>
                    </div>
                </div>
                <div className="flex items-center gap-4">
                    <ThemeToggle />
                    <div className="flex items-center gap-2 px-3 py-1.5 bg-green-500/10 border border-green-500/20 rounded-full">
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-500 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                        </span>
                        <span className="text-[9px] uppercase tracking-[0.2em] font-black text-green-600 dark:text-green-400">Live Insights</span>
                    </div>
                </div>
            </header>

            <main className="p-6 space-y-8 max-w-[1600px] mx-auto w-full">
                {/* ── 1. GLOBAL HEALTH PANEL ────────────────────────────────────── */}
                <section>
                    <div className="flex items-center gap-2 mb-4">
                        <LayoutDashboard className="h-5 w-5 text-[var(--accent)]" />
                        <h2 className="text-lg font-black uppercase tracking-tight">Global Health Layer</h2>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        {Object.entries(globalStats).map(([key, stat]) => (
                            <div key={key} className="bg-[var(--bg-panel)] border border-[var(--border-main)] rounded-xl p-4 hover:border-[var(--accent)] transition-all group shadow-sm">
                                <div className="flex justify-between items-start mb-2">
                                    <span className="text-[10px] uppercase font-black tracking-widest text-[var(--text-muted)]">{METRIC_LABELS[key]}</span>
                                    <span className={`text-[10px] font-mono font-black ${stat.current > 0.7 ? 'text-green-500' : stat.current > 0.5 ? 'text-amber-500' : 'text-red-500'}`}>
                                        {(stat.current * 100).toFixed(1)}%
                                    </span>
                                </div>
                                <div className="flex items-baseline gap-2 mb-4">
                                    <span className="text-2xl font-mono font-black tracking-tighter text-[var(--text-main)]">
                                        {stat.current?.toFixed(4)}
                                    </span>
                                </div>
                                <div className="h-10 w-full opacity-40 group-hover:opacity-100 transition-opacity">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <AreaChart data={stat.trend}>
                                            <Area type="monotone" dataKey="value" stroke={METRIC_COLORS[key]} fill={METRIC_COLORS[key]} fillOpacity={0.1} strokeWidth={2} isAnimationActive={false} />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        ))}
                    </div>
                </section>

                {/* ── 2. TREND & EVOLUTION LAYER ────────────────────────────────── */}
                <section>
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                            <TrendingUp className="h-5 w-5 text-cyan-500" />
                            <h2 className="text-lg font-black uppercase tracking-tight">Trend & Evolution</h2>
                        </div>
                        <div className="flex gap-4">
                            {['mrr', 'recall', 'faithfulness', 'answer_correctness'].map(m => (
                                <div key={m} className="flex items-center gap-1.5 opacity-60 hover:opacity-100 transition-opacity">
                                    <div className="w-2.5 h-2.5 rounded-full shadow-inner" style={{ backgroundColor: METRIC_COLORS[m] }} />
                                    <span className="text-[9px] uppercase font-black tracking-widest text-[var(--text-muted)]">{METRIC_LABELS[m]}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        <div className="bg-[var(--bg-panel)] border border-[var(--border-main)] rounded-xl p-5 h-[350px] shadow-sm">
                            <h3 className="text-[10px] font-black uppercase text-[var(--text-muted)] mb-6 tracking-widest">Main Metrics Progress</h3>
                            <ResponsiveContainer width="100%" height="90%">
                                <LineChart data={trends}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border-main)" vertical={false} />
                                    <XAxis dataKey="timestamp" stroke="var(--text-muted)" fontSize={10} tickLine={false} axisLine={false} dy={10} />
                                    <YAxis stroke="var(--text-muted)" fontSize={10} tickLine={false} axisLine={false} domain={[0, 1]} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: 'var(--bg-panel)', border: '1px solid var(--border-main)', borderRadius: '12px', fontSize: '11px', color: 'var(--text-main)', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)' }}
                                    />
                                    <Line type="monotone" dataKey="mrr_average" stroke={METRIC_COLORS.mrr} strokeWidth={3} dot={{ r: 4, strokeWidth: 0, fill: METRIC_COLORS.mrr }} activeDot={{ r: 6 }} name="MRR" />
                                    <Line type="monotone" dataKey="recall_average" stroke={METRIC_COLORS.recall} strokeWidth={3} dot={{ r: 4, strokeWidth: 0, fill: METRIC_COLORS.recall }} activeDot={{ r: 6 }} name="Recall" />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="bg-[var(--bg-panel)] border border-[var(--border-main)] rounded-xl p-5 h-[350px] shadow-sm">
                            <h3 className="text-[10px] font-black uppercase text-[var(--text-muted)] mb-6 tracking-widest">Quality & Trust Trends</h3>
                            <ResponsiveContainer width="100%" height="90%">
                                <LineChart data={trends}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border-main)" vertical={false} />
                                    <XAxis dataKey="timestamp" stroke="var(--text-muted)" fontSize={10} tickLine={false} axisLine={false} dy={10} />
                                    <YAxis stroke="var(--text-muted)" fontSize={10} tickLine={false} axisLine={false} domain={[0, 1]} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: 'var(--bg-panel)', border: '1px solid var(--border-main)', borderRadius: '12px', fontSize: '11px', color: 'var(--text-main)', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)' }}
                                    />
                                    <Line type="monotone" dataKey="faithfulness_average" stroke={METRIC_COLORS.faithfulness} strokeWidth={3} dot={{ r: 4, strokeWidth: 0, fill: METRIC_COLORS.faithfulness }} name="Faithfulness" />
                                    <Line type="monotone" dataKey="answer_correctness_average" stroke={METRIC_COLORS.answer_correctness} strokeWidth={3} dot={{ r: 4, strokeWidth: 0, fill: METRIC_COLORS.answer_correctness }} name="Correctness" />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </section>

                {/* ── 3. RETRIEVAL VS ANSWER QUALITY MAP ────────────────────────── */}
                <section>
                    <div className="flex items-center gap-2 mb-4">
                        <Target className="h-5 w-5 text-indigo-500" />
                        <h2 className="text-lg font-black uppercase tracking-tight">Retrieval / Quality Alignment</h2>
                    </div>
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        <div className="bg-[var(--bg-panel)] border border-[var(--border-main)] rounded-xl p-5 h-[400px] shadow-sm">
                            <h3 className="text-[10px] font-black uppercase text-[var(--test-muted)] mb-6 tracking-widest">Recall vs Faithfulness</h3>
                            <ResponsiveContainer width="100%" height="90%">
                                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                    <CartesianGrid stroke="var(--border-main)" strokeDasharray="3 3" />
                                    <XAxis type="number" dataKey="recall" name="Recall" stroke="var(--text-muted)" fontSize={10} domain={[0, 1]} label={{ value: 'Recall', position: 'bottom', offset: 0, fontSize: 10, fill: 'var(--text-muted)' }} />
                                    <YAxis type="number" dataKey="faithfulness" name="Faithfulness" stroke="var(--text-muted)" fontSize={10} domain={[0, 1]} label={{ value: 'Faithfulness', angle: -90, position: 'left', offset: 10, fontSize: 10, fill: 'var(--text-muted)' }} />
                                    <Tooltip
                                        cursor={{ strokeDasharray: '3 3' }}
                                        contentStyle={{ backgroundColor: 'var(--bg-panel)', border: '1px solid var(--border-main)', borderRadius: '12px', fontSize: '11px', color: 'var(--text-main)' }}
                                    />
                                    <Scatter name="Questions" data={scatterData}>
                                        {scatterData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={METRIC_COLORS.faithfulness} opacity={0.6} />
                                        ))}
                                    </Scatter>
                                </ScatterChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="bg-[var(--bg-panel)] border border-[var(--border-main)] rounded-xl p-5 h-[400px] shadow-sm">
                            <h3 className="text-[10px] font-black uppercase text-[var(--text-muted)] mb-6 tracking-widest">MRR vs Answer Correctness</h3>
                            <ResponsiveContainer width="100%" height="90%">
                                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                    <CartesianGrid stroke={theme === 'dark' ? '#262626' : '#e5e7eb'} strokeDasharray="3 3" />
                                    <XAxis type="number" dataKey="mrr" name="MRR" stroke={theme === 'dark' ? '#525252' : '#9ca3af'} fontSize={10} domain={[0, 1]} label={{ value: 'MRR', position: 'bottom', offset: 0, fontSize: 10, fill: 'var(--text-muted)' }} />
                                    <YAxis type="number" dataKey="answer_correctness" name="Correctness" stroke={theme === 'dark' ? '#525252' : '#9ca3af'} fontSize={10} domain={[0, 1]} label={{ value: 'Correctness', angle: -90, position: 'left', offset: 10, fontSize: 10, fill: 'var(--text-muted)' }} />
                                    <Tooltip
                                        cursor={{ strokeDasharray: '3 3' }}
                                        contentStyle={{ backgroundColor: 'var(--bg-app)', border: '1px solid var(--border-main)', borderRadius: '8px', fontSize: '11px', color: 'var(--text-main)' }}
                                    />
                                    <Scatter name="Questions" data={scatterData}>
                                        {scatterData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={METRIC_COLORS.answer_correctness} opacity={0.6} />
                                        ))}
                                    </Scatter>
                                </ScatterChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </section>

                {/* ── 4. TOP-K SENSITIVITY CURVE ────────────────────────────────── */}
                <section className="pb-12">
                    <div className="flex items-center gap-2 mb-4">
                        <Activity className="h-5 w-5 text-emerald-500" />
                        <h2 className="text-lg font-black uppercase tracking-tight">K-Factor Sensitivity</h2>
                    </div>
                    <div className="bg-[var(--bg-panel)] border border-[var(--border-main)] rounded-xl p-6 h-[450px] shadow-sm">
                        <div className="flex items-center justify-between mb-8">
                            <div>
                                <h3 className="text-[10px] font-black uppercase text-[var(--text-muted)] mb-1 tracking-widest">Retrieval Depth Optimization</h3>
                                <p className="text-[10px] text-[var(--text-muted)]">Finding the point of diminishing returns for top_k parameter</p>
                            </div>
                        </div>
                        <ResponsiveContainer width="100%" height="80%">
                            <LineChart data={topkData} margin={{ top: 5, right: 30, left: 20, bottom: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-main)" vertical={false} />
                                <XAxis dataKey="top_k" stroke="var(--text-muted)" fontSize={11} label={{ value: 'K-Value (Depth)', position: 'bottom', offset: -10, fontSize: 11, fill: 'var(--text-muted)' }} />
                                <YAxis stroke="var(--text-muted)" fontSize={11} domain={[0, 1]} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: 'var(--bg-panel)', border: '1px solid var(--border-main)', borderRadius: '12px', fontSize: '11px', color: 'var(--text-main)' }}
                                />
                                <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '9px', textTransform: 'uppercase', fontWeight: '900', letterSpacing: '0.1em' }} />
                                <Line type="monotone" dataKey="recall" stroke={METRIC_COLORS.recall} strokeWidth={4} name="Recall" dot={{ r: 5 }} />
                                <Line type="monotone" dataKey="mrr" stroke={METRIC_COLORS.mrr} strokeWidth={2} name="MRR" strokeDasharray="5 5" />
                                <Line type="monotone" dataKey="faithfulness" stroke={METRIC_COLORS.faithfulness} strokeWidth={2} name="Faith" />
                                <Line type="monotone" dataKey="ndcg" stroke={METRIC_COLORS.ndcg} strokeWidth={2} name="NDCG" strokeDasharray="3 3" />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </section>
            </main>
        </div>
    )
}
