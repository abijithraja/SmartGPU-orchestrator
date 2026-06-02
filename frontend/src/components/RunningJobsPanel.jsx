export default function RunningJobsPanel({ jobs }) {
  if (!jobs.length) {
    return <div style={{ color: '#64748b' }}>No running jobs</div>
  }

  return (
    <div>
      {jobs.map((job) => (
        <div
          key={job.job_id}
          style={{
            background: '#0f172a',
            padding: 12,
            borderRadius: 8,
            marginBottom: 12,
            border: '1px solid #334155'
          }}
        >
          <div style={{ marginBottom: 8 }}>
            <strong>{job.model_name}</strong>
          </div>

          <div>GPU: {job.assigned_gpu}</div>

          <div style={{ marginTop: 10 }}>
            <div
              style={{
                height: 10,
                background: '#1e293b',
                borderRadius: 10,
                overflow: 'hidden'
              }}
            >
              <div
                style={{
                  width: `${job.progress || 0}%`,
                  height: '100%',
                  background: '#22c55e',
                  transition: 'width 0.5s ease'
                }}
              />
            </div>
          </div>

          <div style={{ marginTop: 5, fontSize: 13, color: '#94a3b8' }}>
            {job.progress || 0}% Complete
          </div>
        </div>
      ))}
    </div>
  )
}
