export default function RLStatsPanel({ jobs }) {

  const avgConfidence =
    jobs.length
      ? jobs.reduce(
          (a, b) =>
            a + (b.confidence || 0),
          0
        ) / jobs.length
      : 0

  const totalSavings =
    jobs.reduce(
      (acc, j) =>
        acc +
        (
          (j.baseline_cost || 0)
          -
          (j.actual_cost || 0)
        ),
      0
    )

  return (

    <div
      style={{
        display: 'flex',
        gap: 30,
        flexWrap: 'wrap'
      }}
    >

      <div style={{
        background: '#0f172a',
        padding: '12px 20px',
        borderRadius: 8,
        border: '1px solid #334155'
      }}>
        <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 1 }}>
          Avg Confidence
        </div>
        <div style={{ fontSize: 20, fontWeight: 700, color: '#8b5cf6' }}>
          {(avgConfidence * 100).toFixed(1)}%
        </div>
      </div>

      <div style={{
        background: '#0f172a',
        padding: '12px 20px',
        borderRadius: 8,
        border: '1px solid #334155'
      }}>
        <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 1 }}>
          Total Savings
        </div>
        <div style={{ fontSize: 20, fontWeight: 700, color: '#10b981' }}>
          ${totalSavings.toFixed(4)}
        </div>
      </div>

      <div style={{
        background: '#0f172a',
        padding: '12px 20px',
        borderRadius: 8,
        border: '1px solid #334155'
      }}>
        <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 1 }}>
          RL Status
        </div>
        <div style={{ fontSize: 20, fontWeight: 700, color: '#22c55e' }}>
          PPO Active
        </div>
      </div>

    </div>

  )
}
