export default function AIDecisionPanel({ decision }) {
  if (!decision) {
    return (
      <div style={styles.empty}>
        Submit a job to see the AI scheduling decision with explainability.
      </div>
    )
  }

  const savings = decision.baseline_cost_usd && decision.predicted_cost_usd
    ? (decision.baseline_cost_usd - decision.predicted_cost_usd).toFixed(4)
    : null

  const savingsPct = decision.baseline_cost_usd && decision.predicted_cost_usd
    ? (((decision.baseline_cost_usd - decision.predicted_cost_usd) / decision.baseline_cost_usd) * 100).toFixed(1)
    : null

  return (
    <div style={styles.panel}>
      <div style={styles.row}>
        <div style={styles.section}>
          <div style={styles.sectionLabel}>Job ID</div>
          <div style={styles.mono}>{decision.job_id?.slice(0, 8)}...</div>
        </div>
        <div style={styles.section}>
          <div style={styles.sectionLabel}>Assigned GPU</div>
          <div style={{ ...styles.highlight, color: '#818cf8' }}>{decision.assigned_gpu || 'Queued'}</div>
        </div>
        <div style={styles.section}>
          <div style={styles.sectionLabel}>Confidence</div>
          <div style={{ ...styles.highlight, color: '#10b981' }}>
            {decision.confidence ? `${(decision.confidence * 100).toFixed(0)}%` : '--'}
          </div>
        </div>
        <div style={styles.section}>
          <div style={styles.sectionLabel}>Status</div>
          <span style={{ ...styles.statusBadge, background: decision.status === 'running' ? '#1e3a5f' : '#14532d', color: decision.status === 'running' ? '#60a5fa' : '#4ade80' }}>
            {decision.status}
          </span>
        </div>
      </div>

      {savings && (
        <div style={styles.savingsRow}>
          <span style={styles.savingsText}>
            AI saved <strong>${savings}</strong> ({savingsPct}%) vs round-robin baseline
          </span>
          <span>AI: ${decision.predicted_cost_usd?.toFixed(4)} - Baseline: ${decision.baseline_cost_usd?.toFixed(4)}</span>
        </div>
      )}

      {decision.explanation && (
        <div style={styles.explanation}>
          <div style={styles.explainLabel}>Why this GPU?</div>
          <pre style={styles.explainText}>{decision.explanation}</pre>
        </div>
      )}
    </div>
  )
}

const styles = {
  panel: { display: 'flex', flexDirection: 'column', gap: 16 },
  empty: { color: '#64748b', fontSize: 13, padding: '20px 0', textAlign: 'center' },
  row: { display: 'flex', gap: 24, flexWrap: 'wrap' },
  section: {},
  sectionLabel: { fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 4 },
  mono: { fontSize: 13, fontFamily: 'monospace', color: '#94a3b8' },
  highlight: { fontSize: 20, fontWeight: 700 },
  statusBadge: { fontSize: 12, fontWeight: 600, borderRadius: 6, padding: '3px 10px' },
  savingsRow: { background: '#052e16', border: '1px solid #059669', borderRadius: 8, padding: '10px 14px', fontSize: 13, color: '#6ee7b7', display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 },
  savingsText: { color: '#10b981' },
  explanation: { background: '#0f172a', borderRadius: 8, padding: 14, border: '1px solid #334155' },
  explainLabel: { fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 8 },
  explainText: { margin: 0, fontSize: 13, color: '#94a3b8', whiteSpace: 'pre-wrap', fontFamily: 'monospace', lineHeight: 1.6 },
}
