export default function ClusterOverview({ gpus, runningJobs, queuedJobs, totalSavings }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16, marginBottom: 20 }}>
      <StatBox label="GPUs" value={gpus.length} />
      <StatBox label="Workers" value={4} />
      <StatBox label="Running" value={runningJobs.length} />
      <StatBox label="Queued" value={queuedJobs.length} />
      <StatBox label="Savings" value={`$${(totalSavings || 0).toFixed(2)}`} color="var(--green)" />
    </div>
  )
}

function StatBox({ label, value, color = "var(--text)" }) {
  return (
    <div style={{ background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: 4, padding: '12px 16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <div style={{ color: 'var(--muted)', fontSize: 13, fontWeight: 500 }}>{label}</div>
      <div style={{ color: color, fontSize: 20, fontWeight: 600 }}>{value}</div>
    </div>
  )
}
