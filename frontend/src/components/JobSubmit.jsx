import { useState } from 'react'

export default function JobSubmit({ onSubmit, loading }) {
  const [form, setForm] = useState({
    model_name: 'ResNet-50',
    memory_required: 8,
    compute_intensity: 0.7,
    priority: 'normal',
  })

  const handleChange = (e) => {
    const { name, value } = e.target
    setForm(prev => ({
      ...prev,
      [name]: name === 'memory_required' ? parseInt(value) :
               name === 'compute_intensity' ? parseFloat(value) : value
    }))
  }

  const handleSubmit = () => {
    if (!form.model_name.trim()) return alert('Model name required')
    onSubmit(form)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div>
        <label style={styles.label}>Model Name</label>
        <input style={styles.input} name="model_name" value={form.model_name} onChange={handleChange} placeholder="e.g. ResNet-50" />
      </div>
      <div>
        <label style={styles.label}>GPU Memory Required: <strong>{form.memory_required} GB</strong></label>
        <input style={styles.range} type="range" name="memory_required" min={1} max={48} step={1} value={form.memory_required} onChange={handleChange} />
      </div>
      <div>
        <label style={styles.label}>Compute Intensity: <strong>{Math.round(form.compute_intensity * 100)}%</strong></label>
        <input style={styles.range} type="range" name="compute_intensity" min={0} max={1} step={0.1} value={form.compute_intensity} onChange={handleChange} />
      </div>
      <div>
        <label style={styles.label}>Priority</label>
        <select style={styles.input} name="priority" value={form.priority} onChange={handleChange}>
          <option value="low">Low</option>
          <option value="normal">Normal</option>
          <option value="high">High</option>
        </select>
      </div>
      <button style={{ ...styles.btn, opacity: loading ? 0.6 : 1 }} onClick={handleSubmit} disabled={loading}>
        {loading ? 'Scheduling...' : 'Submit Job'}
      </button>
    </div>
  )
}

const styles = {
  label: { display: 'block', fontSize: 12, color: '#94a3b8', marginBottom: 4 },
  input: { width: '100%', background: '#0f172a', border: '1px solid #334155', borderRadius: 6, padding: '8px 10px', color: '#e2e8f0', fontSize: 14, boxSizing: 'border-box' },
  range: { width: '100%', accentColor: '#6366f1' },
  btn: { background: '#6366f1', color: '#fff', border: 'none', borderRadius: 8, padding: '12px', fontSize: 15, fontWeight: 600, cursor: 'pointer', transition: 'opacity 0.2s' },
}
