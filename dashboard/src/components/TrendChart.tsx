import { useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import type { SensorHistoryPoint } from '../types'

interface Props {
  history: SensorHistoryPoint[]
}

type ChartView = 'climate' | 'soil' | 'air'

const VIEWS = [
  {
    key: 'climate' as ChartView,
    label: '온도 / 습도',
    lines: [
      { key: 'temp1', color: '#4DA6FF', name: '온도 (°C)' },
      { key: 'hum1',  color: '#00E5CC', name: '습도 (%RH)' },
    ],
  },
  {
    key: 'soil' as ChartView,
    label: '토양 / 수위',
    lines: [
      { key: 'soilPercent',  color: '#00D084', name: '토양수분 (%)' },
      { key: 'waterPercent', color: '#B066FF', name: '수위 (%)' },
    ],
  },
  {
    key: 'air' as ChartView,
    label: 'CO₂ / 조도',
    lines: [
      { key: 'co2',      color: '#FF9A3C', name: 'CO₂ (ppm)' },
      { key: 'lightRaw', color: '#FFD700', name: '조도 (ADC)' },
    ],
  },
]

interface TooltipProps {
  active?: boolean
  payload?: { color: string; name: string; value: number }[]
  label?: string
}

function CustomTooltip({ active, payload, label }: TooltipProps) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-dark-card border border-dark-border rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-400 mb-1 font-mono">{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }}>{p.name}: <span className="font-bold">{p.value}</span></p>
      ))}
    </div>
  )
}

export default function TrendChart({ history }: Props) {
  const [view, setView] = useState<ChartView>('climate')
  const activeView = VIEWS.find(v => v.key === view)!

  return (
    <section className="px-6 pb-4">
      <div className="card p-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-2">
            <span className="w-3 h-0.5 bg-accent-blue rounded" />
            센서 트렌드 — 최근 20회
          </h2>
          <div className="flex gap-1">
            {VIEWS.map(v => (
              <button
                key={v.key}
                onClick={() => setView(v.key)}
                className={`px-3 py-1 text-xs rounded-md transition-all ${
                  view === v.key
                    ? 'bg-accent-blue/20 text-accent-blue border border-accent-blue/30'
                    : 'text-slate-500 hover:text-slate-300'
                }`}
              >
                {v.label}
              </button>
            ))}
          </div>
        </div>

        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={history} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1A2235" vertical={false} />
            <XAxis
              dataKey="time"
              tick={{ fill: '#475569', fontSize: 10 }}
              tickLine={false}
              axisLine={{ stroke: '#1A2235' }}
              interval={4}
            />
            <YAxis
              tick={{ fill: '#475569', fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              width={36}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: '11px', color: '#94a3b8', paddingTop: '8px' }} />
            {activeView.lines.map(line => (
              <Line
                key={line.key}
                type="monotone"
                dataKey={line.key}
                stroke={line.color}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0 }}
                name={line.name}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
