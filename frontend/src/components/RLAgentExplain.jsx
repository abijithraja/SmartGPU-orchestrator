export default function RLAgentExplain({ decision }) {
  if (!decision) {
    return <div style={styles.empty}>Waiting for PPO decision...</div>
  }

  return (
    <div style={styles.panel}>
      <div style={styles.title}>PPO Reinforcement Learning Agent</div>

      <div style={styles.grid}>
        <div>
          <div style={styles.label}>Selected GPU:</div>
          <div style={styles.value}>{decision.assigned_gpu}</div>
        </div>

        <div>
          <div style={styles.label}>Confidence:</div>
          <div style={styles.value}>
            {typeof decision.confidence === 'number'
              ? `${(decision.confidence * 100).toFixed(1)}%`
              : '--'}
          </div>
        </div>

        <div>
          <div style={styles.label}>RL Cost:</div>
          <div style={styles.value}>
            {typeof decision.actual_cost === 'number'
              ? `$${decision.actual_cost.toFixed(4)}`
              : '--'}
          </div>
        </div>

        <div>
          <div style={styles.label}>Baseline Cost:</div>
          <div style={styles.value}>
            {typeof decision.baseline_cost === 'number'
              ? `$${decision.baseline_cost.toFixed(4)}`
              : '--'}
          </div>
        </div>
      </div>
    </div>
  )
}

const styles = {
  panel: {
    background: '#0f172a',
    borderRadius: 8,
    padding: 16,
  },
  title: {
    color: '#38bdf8',
    marginBottom: 16,
    fontWeight: 700,
  },
  grid: {
    display: 'grid',
    gap: 16,
    gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
  },
  label: {
    fontSize: 12,
    color: '#94a3b8',
    marginBottom: 4,
    textTransform: 'uppercase',
  },
  value: {
    color: '#e2e8f0',
    fontWeight: 600,
  },
  empty: {
    color: '#94a3b8',
    padding: 16,
  },
}
