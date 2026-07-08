
import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import JobSubmit from './components/JobSubmit'
import GPUStatusGrid from './components/GPUStatusGrid'
import AIDecisionPanel from './components/AIDecisionPanel'
import RLAgentExplain from './components/RLAgentExplain'
import RunningJobsPanel from './components/RunningJobsPanel'
import QueuePanel from './components/QueuePanel'
import ComparisonTable from './components/ComparisonTable'
import RLConfidenceChart from './components/RLConfidenceChart'
import GPUHeatMap from './components/GPUHeatMap'
import RLStatsPanel from './components/RLStatsPanel'
import GPUDistribution from './components/GPUDistribution'
import FailurePanel from './components/FailurePanel'
import ClusterHealth from './components/ClusterHealth'
import JobTimeline from './components/JobTimeline'
import ClusterOverview from './components/ClusterOverview'

const API = 'http://localhost:8000'

export default function App() {
  const [gpus, setGpus] = useState([])
  const [jobs, setJobs] = useState([])
  const [runningJobs, setRunningJobs] = useState([])
  const [queuedJobs, setQueuedJobs] = useState([])
  const [clusterMetrics, setClusterMetrics] = useState(null)
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

  const fetchRunningJobs = useCallback(async () => {
    try {
      const res = await axios.get(`${API}/jobs/running`)
      setRunningJobs(res.data)
    } catch (err) {
      console.error('Running jobs fetch error:', err.message)
    }
  }, [])

  const fetchQueuedJobs = useCallback(async () => {
    try {
      const res = await axios.get(`${API}/jobs/queued`)
      setQueuedJobs(res.data)
    } catch (err) {
      console.error('Queued jobs fetch error:', err.message)
    }
  }, [])

  const fetchClusterMetrics = useCallback(async () => {
    try {
      const res = await axios.get(`${API}/api/cluster/metrics`)
      setClusterMetrics(res.data)
    } catch (err) {
      console.error('Cluster metrics fetch error:', err.message)
    }
  }, [])

  useEffect(() => {
    fetchGpus()
    fetchJobs()
    fetchRunningJobs()
    fetchQueuedJobs()
    fetchClusterMetrics()

    // Background polling every 2 seconds for jobs/gpus
    const interval = setInterval(() => {
      fetchGpus()
      fetchJobs()
      fetchRunningJobs()
      fetchQueuedJobs()
    }, 2000)
    
    // Background polling every 5 seconds for cluster metrics
    const metricsInterval = setInterval(() => {
      fetchClusterMetrics()
    }, 5000)
    return () => {
      clearInterval(interval)
      clearInterval(metricsInterval)
    }
  }, [fetchGpus, fetchJobs, fetchRunningJobs, fetchQueuedJobs, fetchClusterMetrics])
  const handleJobSubmit = async (jobData) => {
    setLoading(true)
    try {
      await axios.post(`${API}/jobs/`, jobData)
      
      // Initial fetch to get the 'queued' or 'running' state
      await fetchJobs()
      await fetchRunningJobs()
      await fetchQueuedJobs()

      // Burst polling to catch the worker completion (~5s delay)
      setTimeout(fetchJobs, 2000)
      setTimeout(fetchJobs, 4000)
      setTimeout(fetchJobs, 6000)
      setTimeout(fetchJobs, 8000)
      setTimeout(fetchRunningJobs, 2000)
      setTimeout(fetchQueuedJobs, 2000)
      setTimeout(fetchClusterMetrics, 2000)
      setTimeout(fetchClusterMetrics, 4000)
      setTimeout(fetchClusterMetrics, 6000)

    } catch (err) {
      alert('Job submission failed: ' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  const totalSavings = jobs.reduce((acc, j) => {
    const s = (j.baseline_cost || 0) - (j.actual_cost || 0)
    return acc + (s > 0 ? s : 0)
  }, 0)

  // Extract the latest decision (any status) for live updates
  const latestDecision = jobs
    .sort(
      (a, b) =>
        new Date(b.created_at) -
        new Date(a.created_at)
    )[0]

  return (
    <div style={styles.app}>
      <header style={styles.header}>
        <div>
          <h1 style={styles.title}>SMART GPU ORCHESTRATOR</h1>
          <p style={styles.subtitle}>AI-Driven GPU Scheduling Platform</p>
        </div>
        <div style={{ textAlign: 'right', fontSize: 13, color: 'var(--text)', lineHeight: 1.6 }}>
          <div>Workers: 4</div>
          <div>GPUs: {gpus.length}</div>
          <div style={{ color: 'var(--blue)', fontWeight: 500 }}>Status: Active</div>
        </div>
      </header>

      <div style={styles.grid}>
        <section style={{ gridColumn: '1 / -1' }}>
          <ClusterOverview gpus={gpus} runningJobs={runningJobs} queuedJobs={queuedJobs} totalSavings={totalSavings} />
        </section>

        <section style={{ ...styles.panel, gridColumn: '1 / -1', display: 'flex', gap: 24, flexWrap: 'wrap' }}>
          <div style={{ flex: '2 1 600px' }}>
            <h2 style={styles.cardTitle}>RL Agent Stats</h2>
            <RLStatsPanel jobs={jobs} />
          </div>
          <div style={{ flex: '1 1 300px', background: '#0b0b0b', padding: '0 16px', borderLeft: '1px solid #1f1f1f' }}>
            <h2 style={styles.cardTitle}>Jobs Assigned Distribution</h2>
            <GPUDistribution jobs={jobs} />
          </div>
        </section>

        <section style={styles.panel}>
          <h2 style={styles.cardTitle}>Submit Job</h2>
          <JobSubmit onSubmit={handleJobSubmit} loading={loading} />
        </section>

        <section style={styles.panel}>
          <h2 style={styles.cardTitle}>GPU Status</h2>
          <GPUStatusGrid gpus={clusterMetrics?.gpus || []} />
        </section>

        <section style={styles.panel}>
          <h2 style={styles.cardTitle}>GPU Heat Map</h2>
          <GPUHeatMap gpus={clusterMetrics?.gpus || []} />
        </section>

        <section style={styles.panel}>
          <h2 style={styles.cardTitle}>Cluster Health</h2>
          <ClusterHealth 
            healthScore={clusterMetrics?.cluster_health} 
            gpus={clusterMetrics?.gpus || []} 
          />
        </section>

        <section style={{ ...styles.panel, gridColumn: '1 / -1' }}>
          <AIDecisionPanel decision={latestDecision} />
        </section>

        <section style={{ ...styles.panel, gridColumn: '1 / -1' }}>
          <h2 style={styles.cardTitle}>RL Confidence Trend</h2>
          <RLConfidenceChart jobs={jobs} />
        </section>

        <section style={{ ...styles.panel, gridColumn: '1 / -1' }}>
          <h2 style={styles.cardTitle}>PPO Decision Details</h2>
          <RLAgentExplain decision={latestDecision} />
        </section>

        <section style={styles.panel}>
          <h2 style={styles.cardTitle}>Running Jobs</h2>
          <RunningJobsPanel jobs={runningJobs} />
        </section>

        <section style={styles.panel}>
          <h2 style={styles.cardTitle}>Queued Jobs</h2>
          <QueuePanel jobs={queuedJobs} />
        </section>

        <section style={{ ...styles.panel, gridColumn: '1 / -1' }}>
          <h2 style={styles.cardTitle}>RL vs Round-Robin Comparison</h2>
          <ComparisonTable jobs={jobs} />
        </section>

        <section style={styles.panel}>
          <h2 style={styles.cardTitle}>Failure Recovery</h2>
          <FailurePanel jobs={jobs} />
        </section>

        <section style={styles.panel}>
          <h2 style={styles.cardTitle}>Job Timeline</h2>
          <JobTimeline jobs={jobs} />
        </section>
      </div>
    </div>
  )
}

const styles = {
  app: {
    minHeight: '100vh',
    background: "radial-gradient(circle at top, #111827 0%, #050505 50%, #000000 100%)",
  },
  header: { background: 'var(--panel)', padding: '16px 32px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border)' },
  title: {
    margin: 0,
    fontSize: 18,
    fontWeight: 700,
    color: 'var(--text)',
    letterSpacing: 1
  },
  subtitle: { margin: '4px 0 0', fontSize: 12, color: 'var(--muted)' },
  grid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, padding: '24px 32px' },
  panel: {
    background: "#0b0b0b",
    border: "1px solid #1f1f1f",
    borderRadius: "12px",
    boxShadow: "0 4px 20px rgba(0,0,0,0.35)",
    padding: 20
  },
  cardTitle: {
    margin: '0 0 16px',
    fontSize: 13,
    fontWeight: 600,
    color: "var(--text)",
    textTransform: 'uppercase',
  },
}