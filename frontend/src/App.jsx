import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import JobSubmit from './components/JobSubmit'
import GPUStatusGrid from './components/GPUStatusGrid'
import AIDecisionPanel from './components/AIDecisionPanel'
import ComparisonTable from './components/ComparisonTable'

const API = 'http://localhost:8000'

export default function App() {
  const [gpus, setGpus] = useState([])
  const [jobs, setJobs] = useState([])
  const [lastDecision, setLastDecision] = useState(null)
  const [loading, setLoading] = useState(false)

  const fetchGpus = useCallback(async () => {
    try {
      const res = await axios.get(`${API}/gpus/`)
      setGpus(res.data)
    } catch (err) {
      console.error('GPU fetch error:', err.message)
    }
  }, [])

  const fetchJobs = useCallback(async () => {
    try {
      const res = await axios.get(`${API}/jobs/`)
      setJobs(res.data)
    } catch (err) {
      console.error('Jobs fetch error:', err.message)
    }
  }, [])

  useEffect(() => {
    fetchGpus()
    fetchJobs()
    const interval = setInterval(() => {
      fetchGpus()
      fetchJobs()
    }, 10000)
    return () => clearInterval(interval)
  }, [fetchGpus, fetchJobs])

  const handleJobSubmit = async (jobData) => {
    setLoading(true)
    try {
      const res = await axios.post(`${API}/jobs/`, jobData)
      setLastDecision(res.data)
      await fetchJobs()
    } catch (err) {
      alert('Job submission failed: ' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  const totalSavings = jobs.reduce((acc, j) => {
    const s = (j.baseline_cost_usd || 0) - (j.predicted_cost_usd || 0)
    return acc + (s > 0 ? s : 0)
  }, 0)

  return (
    <div style={styles.app}>
      <header style={styles.header}>
        <div>
          <h1 style={styles.title}>SmartGPU Orchestrator</h1>
          <p style={styles.subtitle}>AI-driven GPU scheduling - PPO reinforcement learning</p>
        </div>
        <div style={styles.savingsBadge}>
          <span style={styles.savingsLabel}>Total Savings</span>
          <span style={styles.savingsValue}>${totalSavings.toFixed(4)}</span>
        </div>
      </header>

      <div style={styles.grid}>
        <section style={styles.card}>
          <h2 style={styles.cardTitle}>Submit Job</h2>
          <JobSubmit onSubmit={handleJobSubmit} loading={loading} />
        </section>

        <section style={styles.card}>
          <h2 style={styles.cardTitle}>GPU Status</h2>
          <GPUStatusGrid gpus={gpus} />
        </section>

        <section style={{ ...styles.card, gridColumn: '1 / -1' }}>
          <h2 style={styles.cardTitle}>AI Decision Panel</h2>
          <AIDecisionPanel decision={lastDecision} />
        </section>

        <section style={{ ...styles.card, gridColumn: '1 / -1' }}>
          <h2 style={styles.cardTitle}>RL vs Round-Robin Comparison</h2>
          <ComparisonTable jobs={jobs} />
        </section>
      </div>
    </div>
  )
}

const styles = {
  app: { minHeight: '100vh', background: '#0f172a', color: '#e2e8f0', fontFamily: 'system-ui, sans-serif', padding: '0 0 40px' },
  header: { background: '#1e293b', padding: '20px 32px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #334155' },
  title: { margin: 0, fontSize: 24, fontWeight: 700, color: '#f1f5f9' },
  subtitle: { margin: '4px 0 0', fontSize: 13, color: '#64748b' },
  savingsBadge: { background: '#064e3b', border: '1px solid #059669', borderRadius: 10, padding: '8px 16px', textAlign: 'center' },
  savingsLabel: { display: 'block', fontSize: 11, color: '#6ee7b7', textTransform: 'uppercase', letterSpacing: 1 },
  savingsValue: { display: 'block', fontSize: 22, fontWeight: 700, color: '#10b981' },
  grid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, padding: '24px 32px' },
  card: { background: '#1e293b', borderRadius: 12, padding: 20, border: '1px solid #334155' },
  cardTitle: { margin: '0 0 16px', fontSize: 16, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: 0.5 },
}
