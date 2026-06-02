import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer
} from 'recharts'

export default function RLConfidenceChart({ jobs }) {

  const data = jobs
    .filter(j => j.confidence)
    .slice(0, 20)
    .reverse()
    .map((j, idx) => ({
      id: idx,
      confidence: Number(
        (j.confidence * 100).toFixed(1)
      )
    }))

  if (!data.length) {
    return (
      <div style={{ color: '#64748b', padding: 16 }}>
        No confidence data yet
      </div>
    )
  }

  return (
    <ResponsiveContainer
      width="100%"
      height={250}
    >
      <LineChart data={data}>
        <XAxis
          dataKey="id"
          stroke="#64748b"
          fontSize={12}
        />
        <YAxis
          domain={[0, 100]}
          stroke="#64748b"
          fontSize={12}
          tickFormatter={(v) => `${v}%`}
        />
        <Tooltip
          contentStyle={{
            background: '#1e293b',
            border: '1px solid #334155',
            borderRadius: 8,
            color: '#e2e8f0'
          }}
          formatter={(v) => [`${v}%`, 'Confidence']}
        />
        <Line
          type="monotone"
          dataKey="confidence"
          stroke="#8b5cf6"
          strokeWidth={2}
          dot={{ fill: '#8b5cf6', r: 4 }}
          activeDot={{ r: 6 }}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
