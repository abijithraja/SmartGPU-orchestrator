export default function ComparisonTable({ jobs }) {
  if (!jobs.length) return <p style={{ color: '#64748b', fontSize: 13 }}>No jobs yet - submit one above.</p>

  const totalSavedUsd = jobs.reduce((acc, j) => {

    const saved =
      (j.baseline_cost || 0) -
      (j.actual_cost || 0)

    return acc + Math.max(saved, 0)

  }, 0)

  return (
    <div>
      <div style={styles.summary}>
        <span>Total jobs: <strong style={{color: '#fff'}}>{jobs.length}</strong></span>
        <span>Cumulative savings: <strong style={{ color: '#10b981' }}>${totalSavedUsd.toFixed(4)}</strong></span>
      </div>
      <style>{`.glass-table-row:hover { background: rgba(59,130,246,0.08) !important; }`}</style>
      <div style={styles.tableWrapper}>
        <table style={styles.table}>
          <thead>
            <tr>
              {['Job ID', 'Model', 'GPU', 'Status', 'RL Cost', 'Baseline', 'Saved', 'Confidence'].map(h => (
                <th key={h} style={styles.th}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {jobs.map(job => {
              const saved = (job.baseline_cost || 0) - (job.actual_cost || 0)
              const pct = job.baseline_cost ? ((saved / job.baseline_cost) * 100).toFixed(0) : 0
              return (
                <tr key={job.job_id} className="glass-table-row" style={styles.tr}>
                  <td style={styles.td}><span style={styles.mono}>{job.job_id?.slice(0, 8)}...</span></td>
                  <td style={styles.td}>{job.model_name}</td>
                  <td style={styles.td}><span style={{ color: '#818cf8' }}>{job.assigned_gpu || '--'}</span></td>
                  <td style={styles.td}>
                    <span style={{ color: job.status === 'completed' ? '#4ade80' : job.status === 'failed' ? '#f87171' : '#facc15' }}>
                      {job.status}
                    </span>
                  </td>
                  <td style={styles.td}>${job.actual_cost?.toFixed(4) ?? '--'}</td>
                  <td style={styles.td}>${job.baseline_cost?.toFixed(4) ?? '--'}</td>
                  <td style={styles.td}>
                    {saved > 0 ? (
                      <span style={{ color: '#10b981' }}>${saved.toFixed(4)} ({pct}%)</span>
                    ) : '--'}
                  </td>
                  <td style={styles.td}>
                    {job.confidence ? `${(job.confidence * 100).toFixed(0)}%` : '--'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const styles = {
  summary: { display: 'flex', gap: 24, fontSize: 13, color: '#888888', marginBottom: 12 },
  tableWrapper: { overflowX: 'auto', background: "rgba(17,17,17,0.75)", borderRadius: "16px", overflow: "hidden" },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 13 },
  th: { textAlign: 'left', padding: '12px', color: '#888888', fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5, borderBottom: '1px solid rgba(255,255,255,0.08)', whiteSpace: 'nowrap' },
  tr: { borderBottom: '1px solid rgba(255,255,255,0.05)', transition: 'background 0.2s ease' },
  td: { padding: '10px 12px', color: '#cbd5e1', verticalAlign: 'middle' },
  mono: { fontFamily: 'monospace', color: '#64748b' },
}
