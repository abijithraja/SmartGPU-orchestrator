export default function GPUHeatMap({ gpus }) {

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: 12
      }}
    >
      {gpus.map(gpu => {

        const util = gpu.utilization

        let color = '#22c55e'

        if (util > 40)
          color = '#f59e0b'

        if (util > 70)
          color = '#ef4444'

        return (
          <div
            key={gpu.id}
            style={{
              background: color,
              borderRadius: 10,
              padding: 20,
              textAlign: 'center',
              fontWeight: 700,
              transition: 'background 0.5s ease'
            }}
          >
            <div style={{ fontSize: 14, marginBottom: 4 }}>{gpu.id}</div>
            <div style={{ fontSize: 22 }}>{util}%</div>
          </div>
        )
      })}
    </div>
  )
}
