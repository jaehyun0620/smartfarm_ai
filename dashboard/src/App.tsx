import Header from './components/Header'
import SystemStatus from './components/SystemStatus'
import SensorGrid from './components/SensorGrid'
import ActuatorGrid from './components/ActuatorGrid'
import TrendChart from './components/TrendChart'
import RecentLog from './components/RecentLog'
import { useSensorData } from './hooks/useSensorData'

export default function App() {
  const {
    mode, setMode,
    sensor,
    history,
    log,
    actuators,
    systemStatus,
    isConnected,
    toggleManual,
  } = useSensorData()

  return (
    <div className="min-h-screen bg-dark-base text-slate-200">
      <div className="max-w-[1600px] mx-auto">
        <Header mode={mode} onModeChange={setMode} isConnected={isConnected} />

        <main className="space-y-0">
          <SystemStatus status={systemStatus} />
          <SensorGrid sensor={sensor} />
          <ActuatorGrid actuators={actuators} mode={mode} onToggle={toggleManual} />
          <TrendChart history={history} />
          <RecentLog log={log} />
        </main>

        <footer className="px-6 py-3 border-t border-dark-border text-center text-xs text-slate-600">
          수동 모드에서는 구동부 카드를 클릭하여 직접 ON/OFF 제어 가능 &nbsp;·&nbsp; 자동 모드에서는 센서값 기반 자동 제어
        </footer>
      </div>
    </div>
  )
}
