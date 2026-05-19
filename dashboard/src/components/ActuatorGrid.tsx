import type { ActuatorState, SystemMode } from '../types'

interface Props {
  actuators: ActuatorState[]
  mode: SystemMode
  onToggle: (id: string) => void
}

function ActuatorCard({ actuator, mode, onToggle }: {
  actuator: ActuatorState
  mode: SystemMode
  onToggle: () => void
}) {
  const isManual = mode === 'manual'
  const isWindow = actuator.id.startsWith('window')

  return (
    <div
      className={`card p-4 flex flex-col gap-2.5 transition-all duration-300 ${
        actuator.isOn ? 'border-accent-green/40 bg-accent-green/5' : 'opacity-70'
      } ${isManual ? 'card-hover cursor-pointer' : ''}`}
      onClick={isManual ? onToggle : undefined}
    >
      {/* Header: icon + toggle */}
      <div className="flex items-center justify-between">
        <span className="text-xl">{actuator.icon}</span>
        <div
          className={`w-9 h-5 rounded-full relative transition-colors duration-300 ${
            actuator.isOn ? 'bg-accent-green' : 'bg-dark-border'
          }`}
        >
          <div
            className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-all duration-300`}
            style={{ left: actuator.isOn ? '1.125rem' : '0.125rem' }}
          />
        </div>
      </div>

      {/* Name */}
      <div>
        <p className="text-sm font-semibold text-white leading-tight">{actuator.name}</p>
        <p className="text-[11px] text-slate-500">{actuator.nameEn}</p>
      </div>

      {/* Status */}
      <div className="flex items-center justify-between gap-1">
        <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${
          actuator.isOn
            ? 'bg-accent-green/15 text-accent-green border border-accent-green/25'
            : 'bg-dark-border text-slate-500'
        }`}>
          {actuator.isOn ? 'ON' : 'OFF'}
        </span>
      </div>

      {/* Window-specific: magnetic + command state */}
      {isWindow && (
        <div className="space-y-0.5">
          <div className="flex items-center justify-between text-[10px]">
            <span className="text-slate-600">자석 감지</span>
            <span className={actuator.magneticState === 'OPEN' ? 'text-accent-green' : 'text-slate-500'}>
              {actuator.magneticState}
            </span>
          </div>
          <div className="flex items-center justify-between text-[10px]">
            <span className="text-slate-600">명령</span>
            <span className={actuator.windowState === 'OPEN' ? 'text-accent-orange' : 'text-slate-500'}>
              {actuator.windowState}
            </span>
          </div>
        </div>
      )}

      {/* Auto reason */}
      {actuator.reason && !isManual && (
        <p className="text-[10px] text-slate-500 truncate" title={actuator.reason}>
          {actuator.reason}
        </p>
      )}

      {isManual && (
        <p className="text-[10px] text-slate-600 text-center mt-auto">클릭하여 제어</p>
      )}
    </div>
  )
}

export default function ActuatorGrid({ actuators, mode, onToggle }: Props) {
  return (
    <section className="px-6 pb-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-2">
          <span className="w-3 h-0.5 bg-accent-orange rounded" />
          구동부 상태
        </h2>
        {mode === 'manual' && (
          <span className="text-xs text-accent-orange bg-accent-orange/10 border border-accent-orange/20 rounded px-2 py-0.5">
            수동 제어 모드 — 카드 클릭으로 ON/OFF
          </span>
        )}
      </div>
      <div className="grid grid-cols-8 gap-3">
        {actuators.map(a => (
          <ActuatorCard
            key={a.id}
            actuator={a}
            mode={mode}
            onToggle={() => onToggle(a.id)}
          />
        ))}
      </div>
    </section>
  )
}
