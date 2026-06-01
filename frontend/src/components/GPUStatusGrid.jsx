import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

function utilColor(val) {
  if (val > 80) return '#ef4444'
  if (val > 50) return '#f59e0b'
  return '#10b981'
}

export default function GPUStatusGrid({ gpus }) {
  if (!gpus.length) return <p style={{ color: '#64748b', fontSize: 13 }}>Loading GPU metrics...</p>

  const chartData = gpus.map(g => ({
    name: g.id,
    utilization: Math.round(g.utilization),
    freeMemory: Math.round(g.free_memory),
    temperature: Math.round(g.temperature),
  }))

  return (
    <div>
      <div style={styles.grid}>
        {gpus.map(gpu => (
          <div key={gpu.id} style={styles.gpuCard}>
            <div style={styles.gpuHeader}>
              <span style={styles.gpuId}>{gpu.id}</span>
              <span style={{ ...styles.badge, background: utilColor(gpu.utilization) + '22', color: utilColor(gpu.utilization), border: `1px solid ${utilColor(gpu.utilization)}44` }}>
                {Math.round(gpu.utilization)}%
              </span>
            </div>
            <div style={styles.metrics}>
              <div><span style={styles.metricLabel}>Free Mem</span><span style={styles.metricVal}>{gpu.free_memory?.toFixed(1)} GB</span></div>
              <div><span style={styles.metricLabel}>Temp</span><span style={styles.metricVal}>{gpu.temperature?.toFixed(0)}C</span></div>
              <div><span style={styles.metricLabel}>Queue</span><span style={styles.metricVal}>{gpu.queue_depth ?? 0}</span></div>
            </div>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 16 }}>
        <ResponsiveContainer width="100%" height={120}>
          <BarChart data={chartData}>
            <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 11 }} />
            <YAxis domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 11 }} />
            <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6, color: '#e2e8f0' }} />
            <Bar dataKey="utilization" fill="#6366f1" radius={[4, 4, 0, 0]} name="Utilization %" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

const styles = {
  grid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 },
  gpuCard: { background: '#0f172a', borderRadius: 8, padding: 12, border: '1px solid #334155' },
  gpuHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  gpuId: { fontSize: 13, fontWeight: 600, color: '#e2e8f0' },
  badge: { fontSize: 12, fontWeight: 700, borderRadius: 6, padding: '2px 8px' },
  metrics: { display: 'flex', gap: 12 },
  metricLabel: { display: 'block', fontSize: 10, color: '#64748b', textTransform: 'uppercase' },
  metricVal: { display: 'block', fontSize: 14, fontWeight: 600, color: '#cbd5e1' },
}
