import type { SystemMode } from '../types'

interface Props {
  mode: SystemMode
  onModeChange: (mode: SystemMode) => void
  isConnected: boolean
}

export default function Header({ mode, onModeChange, isConnected }: Props) {
  const now = new Date()
  const dateStr = now.toISOString().slice(0, 10).replace(/-/g, '-')
  const timeStr = now.toTimeString().slice(0, 5)

  return (
    <header className="border-b border-dark-border px-6 py-4">
      <div className="flex items-start justify-between">
        {/* Left: Title */}
        <div>
          <h1 className="text-2xl font-bold text-white tracking-wide">Smart Farm Monitor</h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Arduino UNO R4 WIFI &nbsp;·&nbsp; Jetson NANO &nbsp;·&nbsp; IoT Capstone
          </p>
        </div>

        {/* Right: Status + Time */}
        <div className="flex items-center gap-4 text-sm">
          <span className="text-slate-400">
            {dateStr} {timeStr}
          </span>
          <span className="text-slate-500 border border-slate-700 rounded px-1.5 py-0.5 text-xs">UTC+9</span>
          <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold ${
            isConnected
              ? 'bg-accent-green/10 text-accent-green border border-accent-green/30'
              : 'bg-red-500/10 text-red-400 border border-red-500/30'
          }`}>
            <span className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-accent-green animate-pulse' : 'bg-red-400'}`} />
            {isConnected ? '연결됨' : '연결 끊김'}
          </div>
        </div>
      </div>

      {/* Mode switcher */}
      <div className="flex items-center justify-between mt-4">
        <div className="flex items-center gap-3 bg-dark-card border border-dark-border rounded-xl px-4 py-2.5">
          <div className="flex items-center gap-2">
            <span className="text-lg">🔄</span>
            <div>
              <p className="text-xs text-slate-500">현재 운전 모드</p>
              <p className="text-sm font-semibold text-white">
                {mode === 'auto' ? '자동 모드' : '수동 모드'}
              </p>
            </div>
          </div>
        </div>

        <div className="flex gap-2">
          <button
            onClick={() => onModeChange('auto')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
              mode === 'auto'
                ? 'bg-accent-blue text-white shadow-lg shadow-accent-blue/20'
                : 'bg-dark-card border border-dark-border text-slate-400 hover:text-white hover:border-accent-blue/40'
            }`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${mode === 'auto' ? 'bg-white animate-pulse' : 'bg-slate-600'}`} />
            자동 모드
          </button>
          <button
            onClick={() => onModeChange('manual')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
              mode === 'manual'
                ? 'bg-accent-orange text-white shadow-lg shadow-accent-orange/20'
                : 'bg-dark-card border border-dark-border text-slate-400 hover:text-white hover:border-accent-orange/40'
            }`}
          >
            <span className="text-base">▶</span>
            수동 모드
          </button>
        </div>
      </div>
    </header>
  )
}
