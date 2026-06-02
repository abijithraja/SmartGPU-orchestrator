export default function JobTimeline({ jobs }) {

  if (!jobs.length) {
    return (
      <div style={{ color: '#64748b', padding: 8 }}>
        No jobs yet
      </div>
    )
  }

  const statusColors = {
    queued: '#f59e0b',
    running: '#3b82f6',
    completed: '#22c55e',
    failed: '#f97316',
    dead: '#ef4444',
  }

  return (
    <div style={{ position: 'relative', paddingLeft: 20 }}>

      {/* Timeline line */}
      <div style={{
        position: 'absolute',
        left: 8,
        top: 0,
        bottom: 0,
        width: 2,
        background: '#334155'
      }} />

      {jobs.slice(0, 10).map(job => (
        <div
          key={job.job_id}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            marginBottom: 14,
            position: 'relative'
          }}
        >
          {/* Timeline dot */}
          <div style={{
            position: 'absolute',
            left: -16,
            width: 10,
            height: 10,
            borderRadius: '50%',
            background: statusColors[job.status] || '#64748b',
            border: '2px solid #1e293b'
          }} />

          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 13, fontWeight: 600 }}>
              {job.model_name}
            </div>
            <div style={{ fontSize: 11, color: '#64748b' }}>
              {job.assigned_gpu || 'pending'} · {job.created_at?.slice(11, 19) || ''}
            </div>
          </div>

          <div style={{
            fontSize: 11,
            fontWeight: 700,
            color: statusColors[job.status] || '#64748b',
            textTransform: 'uppercase'
          }}>
            {job.status}
          </div>
        </div>
      ))}
    </div>
  )
}
