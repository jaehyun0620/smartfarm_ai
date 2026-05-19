import type { SensorData } from '../types'

interface CardDef {
  label: string
  getValue: (s: SensorData) => number
  unit: string
  color: string
  barColor: string
  min: number
  max: number
  normalMin: number
  normalMax: number
  subLabel?: (s: SensorData) => string
}

const CARDS: CardDef[] = [
  {
    label: '온도 (센서1)',
    getValue: s => s.temp1,
    unit: '°C',
    color: 'text-accent-blue',
    barColor: 'bg-accent-blue',
    min: 10, max: 40,
    normalMin: 18, normalMax: 30,
    subLabel: s => `센서2: ${s.temp2.toFixed(1)}°C`,
  },
  {
    label: '습도 (센서1)',
    getValue: s => s.hum1,
    unit: '%RH',
    color: 'text-accent-cyan',
    barColor: 'bg-accent-cyan',
    min: 0, max: 100,
    normalMin: 40, normalMax: 80,
    subLabel: s => `센서2: ${s.hum2.toFixed(1)}%`,
  },
  {
    label: '토양수분',
    getValue: s => s.soilPercent,
    unit: '%',
    color: 'text-accent-green',
    barColor: 'bg-accent-green',
    min: 0, max: 100,
    normalMin: 30, normalMax: 80,
  },
  {
    label: '조도',
    getValue: s => s.lightRaw,
    unit: '',
    color: 'text-accent-yellow',
    barColor: 'bg-accent-yellow',
    min: 0, max: 1023,
    normalMin: 300, normalMax: 950,
    subLabel: () => 'ADC 원시값',
  },
  {
    label: 'CO₂',
    getValue: s => s.co2,
    unit: 'ppm',
    color: 'text-accent-orange',
    barColor: 'bg-accent-orange',
    min: 300, max: 800,
    normalMin: 300, normalMax: 500,
  },
  {
    label: '수위',
    getValue: s => s.waterPercent,
    unit: '%',
    color: 'text-accent-purple',
    barColor: 'bg-accent-purple',
    min: 0, max: 100,
    normalMin: 20, normalMax: 90,
    subLabel: () => '수조 수위',
  },
]

function SensorCard({ def, sensor }: { def: CardDef; sensor: SensorData }) {
  const value = def.getValue(sensor)
  const pct = Math.min(100, Math.max(0, ((value - def.min) / (def.max - def.min)) * 100))
  const isWarning = value < def.normalMin || value > def.normalMax

  return (
    <div className={`card p-4 flex flex-col gap-3 ${isWarning ? 'border-accent-yellow/50' : ''}`}>
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-500 font-medium">{def.label}</span>
        {isWarning && (
          <span className="text-[10px] text-accent-yellow bg-accent-yellow/10 border border-accent-yellow/20 rounded px-1.5 py-0.5">
            주의
          </span>
        )}
      </div>

      <div className="flex items-end gap-1">
        <span className={`text-3xl font-bold tabular-nums leading-none ${def.color}`}>
          {Number.isInteger(value) ? value : value.toFixed(1)}
        </span>
        <span className="text-sm text-slate-500 mb-0.5">{def.unit}</span>
      </div>

      {def.subLabel && (
        <p className="text-[11px] text-slate-600">{def.subLabel(sensor)}</p>
      )}

      <div className="space-y-1">
        <div className="h-1.5 bg-dark-border rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ${isWarning ? 'bg-accent-yellow' : def.barColor}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className="flex justify-between text-[10px] text-slate-600">
          <span>{def.min}</span>
          <span>{def.max}</span>
        </div>
      </div>
    </div>
  )
}

interface Props {
  sensor: SensorData | null
}

export default function SensorGrid({ sensor }: Props) {
  if (!sensor) {
    return (
      <section className="px-6 pb-4">
        <div className="grid grid-cols-6 gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="card h-36 animate-pulse" />
          ))}
        </div>
      </section>
    )
  }

  return (
    <section className="px-6 pb-4">
      <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
        <span className="w-3 h-0.5 bg-accent-cyan rounded" />
        센서 데이터
      </h2>
      <div className="grid grid-cols-6 gap-3">
        {CARDS.map(def => (
          <SensorCard key={def.label} def={def} sensor={sensor} />
        ))}
      </div>
    </section>
  )
}
