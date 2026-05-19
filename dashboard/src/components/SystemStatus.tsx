import type { SystemStatus } from '../types'

interface Props {
  status: SystemStatus | null
}

const stateBadge = {
  NORMAL:  { text: '정상',   cls: 'text-accent-green  bg-accent-green/10  border-accent-green/20' },
  HIGH:    { text: '높음',   cls: 'text-accent-orange bg-accent-orange/10 border-accent-orange/20' },
  LOW:     { text: '낮음',   cls: 'text-accent-blue   bg-accent-blue/10   border-accent-blue/20' },
  KEEP:    { text: '유지',   cls: 'text-accent-cyan   bg-accent-cyan/10   border-accent-cyan/20' },
  ENOUGH:  { text: '충분',   cls: 'text-accent-green  bg-accent-green/10  border-accent-green/20' },
  STABLE:  { text: '안정',   cls: 'text-accent-green  bg-accent-green/10  border-accent-green/20' },
  ACTIVE:  { text: '활성',   cls: 'text-accent-yellow bg-accent-yellow/10 border-accent-yellow/20' },
  MONITOR_ONLY: { text: '모니터링', cls: 'text-slate-400 bg-dark-border border-dark-border' },
}

function Badge({ state }: { state: string }) {
  const cfg = stateBadge[state as keyof typeof stateBadge] ?? {
    text: state, cls: 'text-slate-400 bg-dark-border border-dark-border',
  }
  return (
    <span className={`text-xs font-semibold px-2 py-0.5 rounded-full border ${cfg.cls}`}>
      {cfg.text}
    </span>
  )
}

export default function SystemStatusBar({ status }: Props) {
  if (!status) {
    return (
      <section className="px-6 py-4">
        <div className="h-20 card animate-pulse" />
      </section>
    )
  }

  const stats = [
    { label: '전체 상태',     value: <Badge state={status.overall} />,       sub: status.dayNight === 'DAY' ? '☀️ 주간' : '🌙 야간' },
    { label: '온도 상태',     value: <Badge state={status.tempState} />,      sub: '' },
    { label: '습도 상태',     value: <Badge state={status.humidityState} />,  sub: '' },
    { label: '토양 상태',     value: <Badge state={status.soilState} />,      sub: '' },
    { label: '조도 상태',     value: <Badge state={status.lightState} />,     sub: '' },
    { label: '활성 구동부',   value: <span className="text-2xl font-bold text-accent-blue">{status.activeActuators}</span>, sub: `/ ${status.totalActuators}` },
  ]

  const scenario = status.scenario
    .replace(/\+/g, '  +  ')
    .replace(/_/g, ' ')
    .toLowerCase()

  return (
    <section className="px-6 py-4">
      <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
        <span className="w-3 h-0.5 bg-accent-green rounded" />
        시스템 상태 요약
      </h2>

      <div className="grid grid-cols-6 gap-3 mb-3">
        {stats.map(s => (
          <div key={s.label} className="card px-4 py-3">
            <p className="text-xs text-slate-500 mb-2">{s.label}</p>
            <div className="flex items-end gap-1">
              {s.value}
              {s.sub && <span className="text-xs text-slate-500 mb-0.5 ml-1">{s.sub}</span>}
            </div>
          </div>
        ))}
      </div>

      {/* Scenario banner */}
      <div className="card px-4 py-2 flex items-center gap-2">
        <span className="text-xs text-slate-500 shrink-0">현재 시나리오</span>
        <span className="text-xs text-accent-cyan font-mono truncate">{scenario}</span>
      </div>
    </section>
  )
}
