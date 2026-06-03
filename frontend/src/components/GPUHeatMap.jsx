export default function GPUHeatMap({ gpus }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {gpus.map(gpu => {
        const util = Math.round(gpu.utilization)
        let color = 'var(--green)'
        if (util > 50) color = 'var(--orange)'
        if (util > 80) color = 'var(--red)'
        return (
          <div key={gpu.id} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ width: 50, fontSize: 12, fontWeight: 600, color: 'var(--muted)' }}>{gpu.id.toUpperCase()}</div>
            <div style={{ flex: 1, height: 12, background: 'var(--bg)', border: '1px solid var(--border)' }}>
              <div style={{ width: `${util}%`, background: color, height: '100%' }} />
            </div>
            <div style={{ width: 40, textAlign: 'right', fontSize: 12, color: 'var(--text)' }}>{util}%</div>
          </div>
        )
      })}
    </div>
  )
}
