import type { LogEntry } from '../types'

interface Props {
  log: LogEntry[]
}

export default function RecentLog({ log }: Props) {
  return (
    <section className="px-6 pb-6">
      <div className="card overflow-hidden">
        <div className="px-4 py-3 border-b border-dark-border flex items-center gap-2">
          <span className="w-3 h-0.5 bg-slate-500 rounded" />
          <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">최근 로그</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-dark-border">
                <th className="px-4 py-2 text-left  text-slate-500 font-medium">시간</th>
                <th className="px-4 py-2 text-right text-accent-blue   font-medium">온도</th>
                <th className="px-4 py-2 text-right text-accent-cyan   font-medium">습도</th>
                <th className="px-4 py-2 text-right text-accent-green  font-medium">토양수분</th>
                <th className="px-4 py-2 text-right text-accent-yellow font-medium">조도 (ADC)</th>
                <th className="px-4 py-2 text-right text-accent-orange font-medium">CO₂</th>
                <th className="px-4 py-2 text-right text-accent-purple font-medium">수위</th>
              </tr>
            </thead>
            <tbody>
              {log.map((entry, i) => (
                <tr
                  key={i}
                  className={`border-b border-dark-border/50 transition-colors ${
                    i === 0 ? 'bg-accent-green/5' : 'hover:bg-dark-hover'
                  }`}
                >
                  <td className="px-4 py-2 text-slate-400 font-mono">{entry.time}</td>
                  <td className="px-4 py-2 text-right text-accent-blue">{entry.temp1.toFixed(1)}°C</td>
                  <td className="px-4 py-2 text-right text-accent-cyan">{entry.hum1.toFixed(1)}%</td>
                  <td className="px-4 py-2 text-right text-accent-green">{entry.soilPercent}%</td>
                  <td className="px-4 py-2 text-right text-accent-yellow">{entry.lightRaw}</td>
                  <td className="px-4 py-2 text-right text-accent-orange">{entry.co2} ppm</td>
                  <td className="px-4 py-2 text-right text-accent-purple">{entry.waterPercent}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {log.length === 0 && (
          <p className="text-center text-slate-600 py-6 text-sm">데이터 수집 중...</p>
        )}
      </div>
    </section>
  )
}
