export default function ClusterHealth({ healthScore, gpus }) {
  if (!gpus || !gpus.length) return <div style={{ color: 'var(--text-muted)' }}>Loading...</div>
  const failedCount = gpus.filter(g => g.failed && g.utilization < 1).length
  const score = healthScore ?? Math.max(0, 100 - (failedCount * 25))
  let color = 'var(--green)'
  let label = 'Healthy'
  if (score < 70) { color = 'var(--orange)'; label = 'Warning' }
  if (score < 40) { color = 'var(--red)'; label = 'Critical' }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ fontSize: 48, fontWeight: 700, color: color }}>
        {score}%
      </div>
      <div style={{ fontSize: 16, color: color, fontWeight: 600 }}>
        {label}
      </div>
    </div>
  )
}
