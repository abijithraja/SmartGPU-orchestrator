export default function GPUDistribution({ jobs }) {
  // Count how many times each GPU was assigned
  const distribution = {
    'gpu-0': 0,
    'gpu-1': 0,
    'gpu-2': 0,
    'gpu-3': 0
  }
  
  let totalAssigned = 0;
  
  jobs.forEach(job => {
    if (job.assigned_gpu) {
      if (distribution[job.assigned_gpu] !== undefined) {
        distribution[job.assigned_gpu]++;
        totalAssigned++;
      }
    }
  });
  
  const gpus = Object.keys(distribution);
  
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: -4 }}>
        Total Jobs Scheduled: {totalAssigned}
      </div>
      {gpus.map(gpuId => {
        const count = distribution[gpuId];
        const percent = totalAssigned > 0 ? Math.round((count / totalAssigned) * 100) : 0;
        
        // Distinct colors for each GPU
        const colorMap = {
          'gpu-0': '#3b82f6', // blue
          'gpu-1': '#10b981', // green
          'gpu-2': '#f59e0b', // orange
          'gpu-3': '#8b5cf6'  // purple
        };
        const color = colorMap[gpuId];

        return (
          <div key={gpuId} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ width: 50, fontSize: 12, fontWeight: 600, color: '#cbd5e1' }}>{gpuId.toUpperCase()}</div>
            <div style={{ flex: 1, height: 16, background: '#0f172a', border: '1px solid #1e293b', borderRadius: 4, overflow: 'hidden' }}>
              <div style={{ width: `${percent}%`, background: color, height: '100%', transition: 'width 0.5s ease-in-out' }} />
            </div>
            <div style={{ width: 70, textAlign: 'right', display: 'flex', flexDirection: 'column' }}>
              <span style={{ fontSize: 14, fontWeight: 700, color: '#f8fafc' }}>{count} <span style={{ fontSize: 10, color: '#94a3b8', fontWeight: 500 }}>jobs</span></span>
              <span style={{ fontSize: 11, color: '#64748b' }}>{percent}%</span>
            </div>
          </div>
        )
      })}
    </div>
  )
}
