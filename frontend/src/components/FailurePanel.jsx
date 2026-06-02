export default function FailurePanel({ jobs }) {

  const failed = jobs.filter(
    j => j.status === "failed" || j.status === "dead"
  )

  if (!failed.length) {
    return (
      <div style={{ color: '#64748b', padding: 8 }}>
        No failed jobs — all systems healthy
      </div>
    )
  }

  return (
    <div>
      {failed.map(job => (
        <div
          key={job.job_id}
          style={{
            background: '#0f172a',
            padding: 12,
            borderRadius: 8,
            marginBottom: 10,
            border: '1px solid #7f1d1d',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}
        >
          <div>
            <div style={{ fontWeight: 600 }}>
              {job.model_name}
            </div>
            <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 2 }}>
              {job.job_id?.slice(0, 8)}...
            </div>
          </div>

          <div style={{
            fontSize: 12,
            fontWeight: 700,
            color: job.status === 'dead' ? '#dc2626' : '#f97316',
            textTransform: 'uppercase',
            padding: '4px 10px',
            borderRadius: 6,
            background: job.status === 'dead' ? '#450a0a' : '#431407'
          }}>
            {job.status}
            {job.retry_count > 0 && ` (${job.retry_count} retries)`}
          </div>
        </div>
      ))}
    </div>
  )
}
