export default function QueuePanel({ jobs }) {
  if (!jobs.length) return <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>Queue Empty</div>
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {jobs.map(job => (
        <div key={job.job_id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'var(--panel)', padding: '6px 0', borderBottom: '1px solid var(--border)', fontSize: 13 }}>
          <div style={{ fontWeight: 500, color: 'var(--text)', width: '120px' }}>{job.model_name}</div>
          <div style={{ color: 'var(--text-muted)', flex: 1 }}>{job.priority}</div>
          <div style={{ color: 'var(--text-muted)' }}>Waiting {job.wait_time || 0}s</div>
        </div>
      ))}
    </div>
  )
}
