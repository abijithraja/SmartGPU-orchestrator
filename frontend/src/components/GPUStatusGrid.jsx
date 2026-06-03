export default function GPUStatusGrid({ gpus }) {
  return (
    <div style={styles.grid}>
      {gpus.map(gpu => {
        const util = Math.round(gpu.utilization)
        let color = 'var(--green)'
        if (util > 40 && util <= 70) color = 'var(--blue)'
        if (util > 70 && util <= 90) color = 'var(--orange)'
        if (util > 90) color = 'var(--red)'

        return (
          <div key={gpu.id} style={styles.card}>
            <div style={styles.header}>
              <span style={styles.id}>{gpu.id.toUpperCase()}</span>
            </div>
            
            <div style={styles.barTrack}>
              <div style={{ width: `${util}%`, background: color, height: '100%' }} />
            </div>
            <div style={{ fontSize: 12, color: 'var(--muted)', textAlign: 'right', marginTop: 4 }}>{util}%</div>

            <div style={styles.metrics}>
              <div><span style={styles.label}>Temp</span><span style={styles.val}>{gpu.temperature?.toFixed(0)}°C</span></div>
              <div><span style={styles.label}>Memory</span><span style={styles.val}>{(gpu.memory_used ?? (24 - gpu.free_memory) ?? 0).toFixed(1)} / {gpu.memory_total || 24} GB</span></div>
              <div><span style={styles.label}>Queue</span><span style={styles.val}>{gpu.queue_length ?? gpu.queue_depth ?? 0}</span></div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

const styles = {
  grid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 },
  card: { background: '#0f0f0f', border: '1px solid var(--border)', borderRadius: 8, padding: 16 },
  header: { marginBottom: 12 },
  id: { fontSize: 14, fontWeight: 600, color: 'var(--text)' },
  barTrack: { width: '100%', height: 12, background: 'var(--bg)', border: '1px solid var(--border)' },
  metrics: { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginTop: 12 },
  label: { display: 'block', fontSize: 11, color: 'var(--muted)', marginBottom: 2 },
  val: { display: 'block', fontSize: 13, fontWeight: 500, color: 'var(--text)' },
}
