export default function QueuePanel({ jobs }) {

  if (!jobs.length) {
    return (
      <div style={{ color: '#64748b' }}>
        Queue Empty
      </div>
    )
  }

  return (
    <div>

      {jobs.map(job => (

        <div
          key={job.job_id}
          style={{
            background:'#0f172a',
            padding:12,
            borderRadius:8,
            marginBottom:12,
            border:'1px solid #334155'
          }}
        >
          <div>
            <strong>{job.model_name}</strong>
          </div>

          <div style={{ color: '#94a3b8', fontSize: 13, marginTop: 4 }}>
            Priority: {job.priority}
          </div>

          <div style={{ color: '#f59e0b', fontSize: 13, marginTop: 4 }}>
            Waiting:
            {" "}
            {job.wait_time || 0}
            s
          </div>

        </div>

      ))}

    </div>
  )
}
