export default function RunningJobsPanel({ jobs }) {
  if (!jobs.length) {
    return <div style={styles.empty}>No running jobs</div>
  }

  return (
    <div style={styles.panel}>
      {jobs.map((job) => (
        <div key={job.job_id} style={styles.row}>
          <div>{job.model_name}</div>
          <div>{job.assigned_gpu}</div>
          <div>{job.status}</div>
        </div>
      ))}
    </div>
  )
}

const styles = {
  panel: {
    display: 'grid',
    gap: 10,
  },
  row: {
    display: 'grid',
    gridTemplateColumns: '1.4fr 1fr 0.8fr',
    gap: 12,
    padding: '10px 12px',
    background: '#0f172a',
    borderRadius: 8,
    color: '#e2e8f0',
  },
  empty: {
    color: '#94a3b8',
    padding: 16,
  },
}
