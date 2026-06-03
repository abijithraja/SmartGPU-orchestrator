export default function AIDecisionPanel({ decision }) {
  if (!decision) return <div style={{ color: 'var(--muted)', fontSize: 13 }}>No decisions yet.</div>
  const savingsPct = decision.baseline_cost && decision.cost_saved ? ((decision.cost_saved / decision.baseline_cost) * 100).toFixed(0) : null
  
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text)', borderBottom: '1px solid var(--border)', paddingBottom: 8 }}>
        AI Scheduler Decision
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        <div>
          <div style={styles.lbl}>Selected GPU</div>
          <div style={styles.val}>{decision.assigned_gpu?.toUpperCase() || 'QUEUED'}</div>
        </div>
        <div>
          <div style={styles.lbl}>Confidence</div>
          <div style={styles.val}>{decision.confidence ? `${(decision.confidence * 100).toFixed(0)}%` : '--'}</div>
        </div>
        <div>
          <div style={styles.lbl}>Expected Cost</div>
          <div style={styles.val}>${decision.actual_cost?.toFixed(3) || '--'}</div>
        </div>
        <div>
          <div style={styles.lbl}>Savings</div>
          <div style={{ ...styles.val, color: savingsPct ? 'var(--green)' : 'var(--text)' }}>{savingsPct ? `${savingsPct}%` : '--'}</div>
        </div>
      </div>
      {decision.explanation && (
        <div style={{ marginTop: 8 }}>
          <div style={styles.lbl}>Reason</div>
          <div style={{ fontSize: 13, color: 'var(--text)', marginTop: 4, lineHeight: 1.5 }}>
            {decision.explanation}
          </div>
        </div>
      )}
    </div>
  )
}

const styles = {
  lbl: { fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', marginBottom: 4 },
  val: { fontSize: 16, fontWeight: 600, color: 'var(--text)' }
}