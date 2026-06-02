export default function ClusterHealth({ gpus }) {

  if (!gpus.length) {
    return (
      <div style={{ color: '#64748b' }}>
        Loading cluster data...
      </div>
    )
  }

  const avgUtil =
    gpus.reduce(
      (a, b) => a + b.utilization,
      0
    ) / gpus.length

  const failedCount = gpus.filter(g => g.failed).length

  const score = Math.max(
    0,
    100 - avgUtil - (failedCount * 20)
  )

  let color = '#22c55e'
  let label = 'Excellent'

  if (score < 70) {
    color = '#f59e0b'
    label = 'Moderate'
  }

  if (score < 40) {
    color = '#ef4444'
    label = 'Critical'
  }

  return (
    <div style={{ textAlign: 'center' }}>

      <div style={{
        fontSize: 48,
        fontWeight: 800,
        color: color,
        lineHeight: 1
      }}>
        {score.toFixed(0)}
      </div>

      <div style={{
        fontSize: 13,
        color: color,
        fontWeight: 600,
        marginTop: 4,
        textTransform: 'uppercase',
        letterSpacing: 1
      }}>
        {label}
      </div>

      <div style={{
        fontSize: 11,
        color: '#64748b',
        marginTop: 8
      }}>
        Avg Util: {avgUtil.toFixed(1)}%
        {failedCount > 0 && ` · ${failedCount} GPU(s) offline`}
      </div>

    </div>
  )
}
