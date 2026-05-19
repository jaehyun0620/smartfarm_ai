import { useState, useEffect, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts'

// ── API 설정 ──────────────────────────────────────────────────────────────────
// 백엔드 연결 시 여기만 수정하면 됨
const API_BASE_URL = 'http://localhost:8000'
const API_ENDPOINT = `${API_BASE_URL}/api/sensor-logs`   // GET → records[]
const POLL_INTERVAL_MS = 30_000                           // 30초마다 자동 갱신

// ── State → UI 매핑 ───────────────────────────────────────────────────────────
const STATE_COLOR = {
  NORMAL:       'green',
  STABLE:       'green',
  KEEP:         'green',
  MONITOR_ONLY: 'green',
  HIGH:         'red',
  CRITICAL:     'red',
  LOW:          'orange',
  WARNING:      'orange',
  DRY:          'orange',
}

const STATE_LABEL = {
  NORMAL:       '정상',
  STABLE:       '정상',
  KEEP:         '정상',
  MONITOR_ONLY: '정상',
  HIGH:         '높음',
  CRITICAL:     '위험',
  LOW:          '부족',
  WARNING:      '주의',
  DRY:          '건조',
}

// ── 데이터 변환 ───────────────────────────────────────────────────────────────
// records: 백엔드에서 받은 배열 (오래된 순 정렬)
// 반환: { sensors, statusBars, deviceStates, chartData, logRows, plantSummary, latest }
function transformRecords(records) {
  if (!records || records.length === 0) return null

  const latest = records[records.length - 1]

  // 센서 값 계산
  const temp     = parseFloat(latest.temp1)
  const hum      = parseFloat(latest.hum1)
  const soil     = parseInt(latest.soil_percent)
  const lightRaw = parseInt(latest.light_raw)
  const co2      = parseInt(latest.co2_raw)
  const waterPct = Math.min(100, Math.round(parseInt(latest.water_raw) / 1023 * 100))

  // ── 센서 카드 데이터 ──
  const sensors = {
    soil:    { value: soil,    unit: '%',   label: '토양 수분', range: '적정 40~70%',      status: soil >= 40 && soil <= 70 ? '정상' : soil < 40 ? '부족' : '과습',          color: soil >= 40 && soil <= 70 ? 'green' : soil < 40 ? 'orange' : 'blue' },
    co2:     { value: co2,     unit: 'ppm', label: 'CO₂',      range: '적정 400~1000ppm', status: STATE_LABEL[latest.co2_state]      ?? '정상', color: STATE_COLOR[latest.co2_state]      ?? 'green' },
    temp:    { value: temp,    unit: '°C',  label: '온도',      range: '적정 18~28°C',     status: STATE_LABEL[latest.temp_state]     ?? '정상', color: STATE_COLOR[latest.temp_state]     ?? 'green' },
    humidity:{ value: hum,     unit: '%',   label: '습도',      range: '적정 50~70%',      status: STATE_LABEL[latest.humidity_state] ?? '정상', color: STATE_COLOR[latest.humidity_state] ?? 'green' },
    light:   { value: lightRaw,unit: 'lx',  label: '조도',      range: '적정 500~2000lx',  status: STATE_LABEL[latest.light_state]    ?? '정상', color: STATE_COLOR[latest.light_state]    ?? 'green' },
    water:   { value: waterPct,unit: '%',   label: '수위',      range: '주의 20% 이하',    status: waterPct > 20 ? '충분' : '부족',                                           color: waterPct > 20 ? 'blue' : 'red' },
  }

  // ── 홈 상태바 ──
  const co2Fill  = co2 < 800 ? 80 : co2 < 1200 ? 40 : 20
  const co2Color = co2 < 800 ? 'green' : co2 < 1200 ? 'orange' : 'red'
  const lightFill = Math.min(95, Math.round(lightRaw / 20))
  const thOk     = latest.temp_state === 'NORMAL' && latest.humidity_state === 'NORMAL'

  const statusBars = [
    {
      key: 'soil', icon: '💧', label: '수분',
      fill: soil, color: soil >= 40 && soil <= 70 ? 'green' : soil < 40 ? 'orange' : 'blue',
      text: soil >= 40 && soil <= 70 ? '좋아요' : soil < 40 ? '부족해요' : '과습해요',
      detail: soil < 40
        ? '흙이 바짝 말랐어요. 지금 바로 물을 줘야 해요.'
        : soil > 70
        ? '흙이 너무 축축해요. 물을 잠시 멈춰주세요.'
        : '흙 상태가 좋아요. 현재 수분을 유지해 주세요.',
    },
    {
      key: 'air', icon: '☴', label: '공기',
      fill: co2Fill, color: co2Color,
      text: co2 < 800 ? '맑아요' : co2 < 1200 ? '주의' : '탁해요',
      detail: co2 < 800
        ? '공기가 맑아요. 환기 상태가 좋습니다.'
        : co2 < 1200
        ? '공기가 조금 탁해요. 환기를 권장합니다.'
        : '숨쉬기 답답한 상태에요. AI가 환기팬을 켰어요. 창문도 열어주시면 훨씬 빨리 좋아져요.',
    },
    {
      key: 'sun', icon: '☀️', label: '햇빛',
      fill: lightFill, color: STATE_COLOR[latest.light_state] ?? 'green',
      text: latest.light_state === 'NORMAL' ? '딱 좋아요' : latest.light_state === 'LOW' ? '어두워요' : '너무 밝아요',
      detail: latest.light_state === 'NORMAL'
        ? '지금 밝기가 딱 맞아요. 이대로 유지해 주세요.'
        : latest.light_state === 'LOW'
        ? '빛이 부족해요. LED를 켜거나 창 가까이 옮겨주세요.'
        : '빛이 너무 강해요. 직사광선을 피해주세요.',
    },
    {
      key: 'th', icon: '🌡', label: '온도·습도',
      fill: thOk ? 80 : 45, color: thOk ? 'green' : 'orange',
      text: thOk ? '쾌적해요' : '주의',
      detail: thOk
        ? '따로 신경 쓰지 않아도 괜찮아요. AI가 관리하고 있어요.'
        : `온도 ${latest.temp_state === 'NORMAL' ? '정상' : latest.temp_state} / 습도 ${latest.humidity_state === 'NORMAL' ? '정상' : latest.humidity_state}. 조절이 필요해요.`,
    },
    {
      key: 'water', icon: '≋', label: '물통 수위',
      fill: waterPct, color: 'blue',
      text: waterPct > 50 ? '충분해요' : waterPct > 20 ? '절반 남았어요' : '부족해요',
      detail: waterPct > 50
        ? '물통에 물이 충분해요. 약 2주 분량이 남았어요.'
        : waterPct > 20
        ? '물통 물이 절반 정도 남았어요. 곧 보충해 주세요.'
        : '물통이 거의 비었어요. 지금 바로 채워주세요.',
    },
  ]

  // ── 기기 상태 (autoMode용 초기값) ──
  const isOn = (v) => v === '1' || v === 1
  const deviceStates = [
    isOn(latest.fan1),
    isOn(latest.fan2),
    isOn(latest.window1),
    isOn(latest.window2),
    isOn(latest.led),
    isOn(latest.humidifier),
    isOn(latest.heater),
  ]

  // ── 차트 데이터 ──
  const toTime = (ts) => {
    const d = new Date(ts)
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
  }
  const chartData = {
    soil:     records.map(r => ({ t: toTime(r.timestamp), v: parseInt(r.soil_percent) })),
    co2:      records.map(r => ({ t: toTime(r.timestamp), v: parseInt(r.co2_raw) })),
    temp:     records.map(r => ({ t: toTime(r.timestamp), v: parseFloat(r.temp1) })),
    humidity: records.map(r => ({ t: toTime(r.timestamp), v: parseFloat(r.hum1) })),
    light:    records.map(r => ({ t: toTime(r.timestamp), v: parseInt(r.light_raw) })),
    water:    records.map(r => ({ t: toTime(r.timestamp), v: Math.min(100, Math.round(parseInt(r.water_raw) / 1023 * 100)) })),
  }

  // ── 센서 기록 테이블 (최근 10개, 최신순) ──
  const logRows = [...records].reverse().slice(0, 10).map(r => ({
    time:     toTime(r.timestamp),
    temp:     `${parseFloat(r.temp1).toFixed(1)}°C`,
    hum:      `${parseFloat(r.hum1).toFixed(0)}%`,
    soil:     `${r.soil_percent}%`,
    soilNum:  parseInt(r.soil_percent),
    co2:      parseInt(r.co2_raw),
    light:    parseInt(r.light_raw),
    water:    `${Math.min(100, Math.round(parseInt(r.water_raw) / 1023 * 100))}%`,
  }))

  // ── 홈 식물 요약 ──
  const issues = []
  if (soil < 40)  issues.push('물이 부족해요')
  if (co2 > 1000) issues.push('공기가 탁해요')
  if (latest.temp_state     !== 'NORMAL') issues.push('온도가 불안정해요')
  if (latest.humidity_state !== 'NORMAL') issues.push('습도를 조절해야 해요')

  const plantSummary = {
    title:    issues.length === 0 ? '잘 자라고 있어요' : issues.length === 1 ? '조금 신경 써야 해요' : '조금 힘들어하고 있어요',
    subtitle: issues.length === 0 ? '모든 상태가 좋아요' : issues.join(', '),
  }

  return { sensors, statusBars, deviceStates, chartData, logRows, plantSummary, latest }
}

// ── Mock 데이터 (백엔드 연결 전 fallback) ─────────────────────────────────────
const MOCK_RECORDS = Array.from({ length: 10 }, (_, i) => {
  const d = new Date('2026-05-15T15:58:50')
  d.setMinutes(d.getMinutes() - (9 - i) * 2)
  return {
    timestamp:        d.toISOString().slice(0, 19),
    overall_state:    'WARNING',
    temp_state:       'NORMAL',
    humidity_state:   'NORMAL',
    soil_state:       'LOW',
    light_state:      'NORMAL',
    co2_state:        'HIGH',
    temp1:            (22 + i * 0.1).toFixed(1),
    hum1:             (61 + i * 0.1).toFixed(1),
    temp2:            (24 + i * 0.1).toFixed(1),
    hum2:             (63 + i * 0.1).toFixed(1),
    soil_percent:     String(28 + i * 2),
    light_raw:        String(820 + i * 4),
    water_raw:        '815',
    co2_raw:          String(1420 - i * 40),
    fan1:             '1', fan2: '1',
    window1:          '0', window2: '0',
    led:              '1', humidifier: '0', heater: '0', pump: '0',
    window1_magnetic: 'CLOSED', window2_magnetic: 'CLOSED',
    window1_command:  'CLOSED', window2_command:  'CLOSED',
    window1_sensor:   'CLOSED', window2_sensor:   'CLOSED',
    scenario:         'CO2_HIGH_VENTILATION',
  }
})

// ── 탭 / 기기 정의 ────────────────────────────────────────────────────────────
const TABS = [
  { key: 'soil',     label: '토양 수분' },
  { key: 'co2',      label: 'CO₂'      },
  { key: 'temp',     label: '온도'      },
  { key: 'humidity', label: '습도'      },
  { key: 'light',    label: '조도'      },
  { key: 'water',    label: '수위'      },
]

const DEVICE_META = [
  { icon: '☴',  name: '환기팬 1', sub: '흡기', type: 'on-off',    autoDescOn: '신선한 공기를 안으로 들이는 중', autoDescOff: '지금은 환기가 필요 없어요' },
  { icon: '☴',  name: '환기팬 2', sub: '배기', type: 'on-off',    autoDescOn: '탁한 공기를 밖으로 내보내는 중', autoDescOff: '지금은 환기가 필요 없어요' },
  { icon: '▬',  name: '창문 1',   sub: '앞면', type: 'open-close',autoDescOn: '환기를 위해 열어뒀어요',          autoDescOff: '지금은 닫아두는 게 더 나아요' },
  { icon: '▬',  name: '창문 2',   sub: '뒷면', type: 'open-close',autoDescOn: '환기를 위해 열어뒀어요',          autoDescOff: '지금은 닫아두는 게 더 나아요' },
  { icon: '💡', name: '식물 LED', sub: '',     type: 'on-off',    autoDescOn: '잘 자랄 수 있게 빛을 켜줬어요',   autoDescOff: '지금은 자연광으로 충분해요' },
  { icon: '☁',  name: '가습기',   sub: '',     type: 'on-off',    autoDescOn: '습도를 높이는 중이에요',           autoDescOff: '지금 습도가 딱 좋아서 쉬는 중' },
  { icon: '🔥', name: '히터',     sub: '',     type: 'on-off',    autoDescOn: '온도를 높이는 중이에요',           autoDescOff: '지금 온도가 딱 좋아서 쉬는 중' },
]

// ── 색상 팔레트 ───────────────────────────────────────────────────────────────
const COLOR = {
  orange: { text: '#d97706', bar: '#f97316', bg: '#fffbeb' },
  red:    { text: '#dc2626', bar: '#ef4444', bg: '#fef2f2' },
  green:  { text: '#16a34a', bar: '#22c55e', bg: '#f0fdf4' },
  blue:   { text: '#2563eb', bar: '#3b82f6', bg: '#eff6ff' },
}

function chartColor(tab) {
  return { soil: '#f97316', co2: '#ef4444', temp: '#22c55e', humidity: '#22c55e', light: '#22c55e', water: '#3b82f6' }[tab] ?? '#6b7280'
}

// ── Home page ─────────────────────────────────────────────────────────────────
function HomePage({ data, onGoSensor }) {
  const [autoMode, setAutoMode] = useState(true)
  const [watered, setWatered]   = useState(false)
  const [expanded, setExpanded] = useState(null)

  // 기기 상태: 백엔드 데이터로 초기화, 수동 모드에서 사용자가 변경 가능
  const [devOnState, setDevOnState] = useState(() => data?.deviceStates ?? [true, true, false, false, true, false, false])
  useEffect(() => {
    if (data?.deviceStates) setDevOnState(data.deviceStates)
  }, [data])

  const toggleDevice = (idx, val) => {
    setDevOnState(prev => prev.map((v, i) => i === idx ? val : v))
    // TODO: POST `${API_BASE_URL}/api/control` with { device: DEVICE_META[idx].name, state: val }
  }

  const { statusBars, plantSummary } = data ?? {
    statusBars:   [],
    plantSummary: { title: '불러오는 중…', subtitle: '' },
  }

  const needsWater = data?.latest && parseInt(data.latest.soil_percent) < 40
  const needsVent  = data?.latest && parseInt(data.latest.co2_raw) > 1000
  const fan1On     = devOnState[0]
  const fan2On     = devOnState[1]

  return (
    <div style={{ maxWidth: 480, margin: '0 auto', padding: '0 0 40px' }}>

      {/* ── Plant Header ── */}
      <div style={{ padding: '20px 20px 0', display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{ width: 48, height: 48, borderRadius: '50%', background: '#dcfce7', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 24 }}>🌿</div>
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 700, fontSize: 17 }}>홍콩야자</div>
          <div style={{ color: '#6b7280', fontSize: 13 }}>거실 창가 · 37일째</div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: '#22c55e' }} />
          <span style={{ color: '#22c55e', fontSize: 12, fontWeight: 600 }}>실시간</span>
        </div>
      </div>

      {/* ── Plant Status Card ── */}
      <div style={{ margin: '16px 16px 0', background: '#fff', borderRadius: 16, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.08)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
          <div style={{ width: 40, height: 40, borderRadius: '50%', background: '#dcfce7', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 20 }}>🌿</div>
          <div>
            <div style={{ color: '#6b7280', fontSize: 12, marginBottom: 2 }}>오늘 식물 상태</div>
            <div style={{ fontWeight: 700, fontSize: 18 }}>{plantSummary.title}</div>
            <div style={{ color: '#6b7280', fontSize: 13 }}>{plantSummary.subtitle}</div>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {statusBars.map(({ key, icon, label, fill, color, text, detail }) => {
            const isOpen = expanded === key
            return (
              <div key={key}>
                <div
                  onClick={() => setExpanded(isOpen ? null : key)}
                  style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '9px 0', cursor: 'pointer' }}
                >
                  <span style={{ fontSize: 16, width: 22, textAlign: 'center' }}>{icon}</span>
                  <span style={{ fontSize: 13, color: '#374151', width: 72, flexShrink: 0 }}>{label}</span>
                  <div style={{ flex: 1, height: 6, background: '#f3f4f6', borderRadius: 99, overflow: 'hidden' }}>
                    <div style={{ width: `${fill}%`, height: '100%', background: COLOR[color].bar, borderRadius: 99, transition: 'width 0.6s ease' }} />
                  </div>
                  <span style={{ fontSize: 12, color: COLOR[color].text, width: 64, textAlign: 'right', fontWeight: 500 }}>{text}</span>
                </div>
                {isOpen && (
                  <div style={{ margin: '0 0 8px 32px', padding: '10px 12px', background: COLOR[color].bg, borderRadius: 10, fontSize: 13, color: COLOR[color].text, lineHeight: 1.5 }}>
                    {detail}
                  </div>
                )}
              </div>
            )
          })}
        </div>

        <div style={{ marginTop: 8, color: '#9ca3af', fontSize: 12, textAlign: 'center' }}>항목을 누르면 이유를 알 수 있어요</div>
      </div>

      {/* ── 지금 해주세요 ── */}
      <div style={{ padding: '20px 16px 0' }}>
        <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 10 }}>지금 해주세요</div>

        {needsWater && (
          <div style={{ background: '#fffbeb', borderRadius: 14, padding: '14px 16px', display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
            <span style={{ fontSize: 22 }}>🤚</span>
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 600, fontSize: 14 }}>직접 물을 줘야 해요</div>
              <div style={{ color: '#78716c', fontSize: 12, marginTop: 2 }}>흙이 바짝 말랐어요 · 500ml 정도 주세요</div>
            </div>
            <button
              onClick={() => setWatered(v => !v)}
              style={{ background: watered ? '#22c55e' : '#1c1c1e', color: '#fff', border: 'none', borderRadius: 20, padding: '6px 14px', fontSize: 13, fontWeight: 600, cursor: 'pointer', whiteSpace: 'nowrap', fontFamily: 'inherit' }}
            >
              {watered ? '✓ 줬어요' : '줬어요 ✓'}
            </button>
          </div>
        )}

        {(needsVent || fan1On || fan2On) && (
          <div style={{ background: '#eff6ff', borderRadius: 14, padding: '14px 16px', display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 22 }}>☴</span>
            <div>
              <div style={{ fontWeight: 600, fontSize: 14, color: '#1d4ed8' }}>AI가 환기를 켰어요</div>
              <div style={{ color: '#6b7280', fontSize: 12, marginTop: 2 }}>창문도 열어주시면 공기가 더 빨리 맑아져요</div>
            </div>
          </div>
        )}

        {!needsWater && !needsVent && !fan1On && !fan2On && (
          <div style={{ background: '#f0fdf4', borderRadius: 14, padding: '14px 16px', display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 22 }}>✅</span>
            <div>
              <div style={{ fontWeight: 600, fontSize: 14, color: '#16a34a' }}>AI가 알아서 관리 중이에요</div>
              <div style={{ color: '#6b7280', fontSize: 12, marginTop: 2 }}>지금은 특별히 해야 할 일이 없어요</div>
            </div>
          </div>
        )}
      </div>

      {/* ── AI / 수동 제어 섹션 ── */}
      <div style={{ padding: '20px 16px 0' }}>
        <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 10 }}>
          {autoMode ? 'AI가 알아서 하고 있어요' : '직접 제어 중'}
        </div>

        {!autoMode && (
          <div style={{ background: '#fff7ed', borderRadius: 14, padding: '14px 16px', marginBottom: 10, display: 'flex', alignItems: 'flex-start', gap: 10, border: '1px solid #fed7aa' }}>
            <span style={{ fontSize: 18 }}>⚠️</span>
            <div>
              <div style={{ fontWeight: 600, fontSize: 14, color: '#c2410c' }}>AI 자동 제어가 꺼져 있어요</div>
              <div style={{ color: '#9a3412', fontSize: 12, marginTop: 2 }}>구동부를 직접 조작하고 있어요 · 주의해서 사용하세요</div>
            </div>
          </div>
        )}

        <div style={{ background: '#fff', borderRadius: 16, overflow: 'hidden', boxShadow: '0 1px 4px rgba(0,0,0,0.08)' }}>

          {/* 자동/수동 토글 */}
          <div style={{ display: 'flex', alignItems: 'center', padding: '14px 16px', borderBottom: '1px solid #f3f4f6', background: autoMode ? '#fff' : '#fff7ed' }}>
            <span style={{ fontSize: 18, marginRight: 10 }}>{autoMode ? '🤖' : '🔧'}</span>
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 600, fontSize: 14 }}>{autoMode ? '자동 모드' : '수동 모드'}</div>
              <div style={{ color: '#9ca3af', fontSize: 12 }}>{autoMode ? 'AI가 센서를 보고 알아서 제어해요' : 'AI 자동 제어 꺼짐 · 직접 조작하세요'}</div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: 12, color: autoMode ? '#16a34a' : '#9ca3af', fontWeight: 500 }}>자동</span>
              <div onClick={() => setAutoMode(v => !v)} style={{ width: 44, height: 24, borderRadius: 12, cursor: 'pointer', position: 'relative', background: autoMode ? '#22c55e' : '#f97316', transition: 'background 0.2s' }}>
                <div style={{ position: 'absolute', top: 2, left: autoMode ? 22 : 2, width: 20, height: 20, borderRadius: '50%', background: '#fff', boxShadow: '0 1px 3px rgba(0,0,0,0.2)', transition: 'left 0.2s' }} />
              </div>
              <span style={{ fontSize: 12, color: autoMode ? '#9ca3af' : '#c2410c', fontWeight: 500 }}>수동</span>
            </div>
          </div>

          {/* 기기 목록 */}
          {DEVICE_META.map((meta, i) => {
            const isOn   = devOnState[i] ?? false
            const onLbl  = meta.type === 'open-close' ? '열기' : '켜기'
            const offLbl = meta.type === 'open-close' ? '닫기' : '끄기'
            const statusLbl  = meta.type === 'open-close' ? (isOn ? '열림' : '닫힘') : (isOn ? '켜짐' : '대기')
            const badgeBg    = (statusLbl === '켜짐' || statusLbl === '열림') ? '#dcfce7' : '#f3f4f6'
            const badgeText  = (statusLbl === '켜짐' || statusLbl === '열림') ? '#16a34a' : '#9ca3af'
            const autoDesc   = isOn ? meta.autoDescOn : meta.autoDescOff
            const manualDesc = isOn ? `사용자가 ${onLbl.slice(0, -1)}줬어요` : `사용자가 ${offLbl}어요`

            return (
              <div key={i} style={{ display: 'flex', alignItems: 'center', padding: '13px 16px', borderBottom: i < DEVICE_META.length - 1 ? '1px solid #f3f4f6' : 'none' }}>
                <span style={{ fontSize: 18, marginRight: 10, width: 24, textAlign: 'center' }}>{meta.icon}</span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 14, fontWeight: 500 }}>
                    {meta.name}
                    {meta.sub && <span style={{ color: '#9ca3af', fontWeight: 400, fontSize: 12, marginLeft: 4 }}>{meta.sub}</span>}
                  </div>
                  <div style={{ color: '#9ca3af', fontSize: 12, marginTop: 1 }}>{autoMode ? autoDesc : manualDesc}</div>
                </div>
                {autoMode ? (
                  <span style={{ fontSize: 12, fontWeight: 600, padding: '3px 10px', borderRadius: 20, background: badgeBg, color: badgeText }}>{statusLbl}</span>
                ) : (
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button onClick={() => toggleDevice(i, true)}  style={{ padding: '5px 12px', borderRadius: 20, border: 'none', fontSize: 12, fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit', background: isOn  ? '#22c55e' : '#f3f4f6', color: isOn  ? '#fff' : '#6b7280', transition: 'background 0.15s' }}>{onLbl}</button>
                    <button onClick={() => toggleDevice(i, false)} style={{ padding: '5px 12px', borderRadius: 20, border: 'none', fontSize: 12, fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit', background: !isOn ? '#22c55e' : '#f3f4f6', color: !isOn ? '#fff' : '#6b7280', transition: 'background 0.15s' }}>{offLbl}</button>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* ── 센서 페이지 링크 ── */}
      <div style={{ padding: '16px 16px 0' }}>
        <button onClick={onGoSensor} style={{ width: '100%', background: '#fff', border: '1px solid #e5e7eb', borderRadius: 14, padding: '16px 20px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, fontFamily: 'inherit', fontWeight: 600, fontSize: 15, color: '#1c1c1e' }}>
          <span>📊</span><span>센서 수치 · 그래프 · 로그 보기</span><span style={{ color: '#9ca3af' }}>→</span>
        </button>
      </div>
    </div>
  )
}

// ── Sensor page ───────────────────────────────────────────────────────────────
function SensorPage({ data, onBack }) {
  const [activeTab, setActiveTab] = useState('soil')

  const sensors   = data?.sensors   ?? {}
  const chartData = data?.chartData ?? {}
  const logRows   = data?.logRows   ?? []
  const s         = sensors[activeTab] ?? { label: '', range: '', value: '-', unit: '', status: '', color: 'green' }
  const warning   = activeTab === 'soil' && s.status === '부족' ? '부족 — 물주기 필요'
                  : activeTab === 'co2'  && s.status === '높음' ? '높음 — 환기 필요'
                  : null

  return (
    <div style={{ maxWidth: 480, margin: '0 auto', padding: '0 0 40px' }}>

      <div style={{ padding: '20px 16px 0', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
        <div>
          <div style={{ fontWeight: 700, fontSize: 20 }}>센서 수치 · 로그</div>
          <div style={{ color: '#6b7280', fontSize: 13, marginTop: 2 }}>홍콩야자 · 오늘 데이터</div>
        </div>
        <button onClick={onBack} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#6b7280', fontSize: 14, fontFamily: 'inherit', padding: '4px 0' }}>← 홈으로</button>
      </div>

      {/* 탭 */}
      <div style={{ padding: '14px 16px 0', display: 'flex', gap: 8, overflowX: 'auto', paddingBottom: 2 }}>
        {TABS.map(({ key, label }) => (
          <button key={key} onClick={() => setActiveTab(key)} style={{ flexShrink: 0, padding: '7px 14px', borderRadius: 99, border: 'none', cursor: 'pointer', fontFamily: 'inherit', fontSize: 13, fontWeight: 500, background: activeTab === key ? '#1c1c1e' : '#e5e7eb', color: activeTab === key ? '#fff' : '#374151', transition: 'background 0.15s' }}>{label}</button>
        ))}
      </div>

      {/* 차트 카드 */}
      <div style={{ margin: '14px 16px 0', background: '#fff', borderRadius: 16, padding: '18px 18px 14px', boxShadow: '0 1px 4px rgba(0,0,0,0.08)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 4 }}>
          <div>
            <div style={{ fontWeight: 700, fontSize: 16 }}>{s.label}</div>
            <div style={{ color: '#9ca3af', fontSize: 12 }}>{s.range}</div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <span style={{ fontSize: 28, fontWeight: 700, color: COLOR[s.color]?.text }}>{s.value}</span>
            <span style={{ fontSize: 13, color: '#9ca3af', marginLeft: 2 }}>{s.unit}</span>
            <div style={{ color: '#9ca3af', fontSize: 11, marginTop: 1 }}>현재 수치</div>
          </div>
        </div>
        {warning && (
          <div style={{ display: 'inline-block', marginBottom: 10, background: COLOR[s.color]?.bg, color: COLOR[s.color]?.text, fontSize: 12, fontWeight: 500, padding: '4px 10px', borderRadius: 8 }}>{warning}</div>
        )}
        <div style={{ height: 120 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData[activeTab] ?? []} margin={{ top: 4, right: 4, left: -28, bottom: 0 }}>
              <XAxis dataKey="t" tick={{ fontSize: 10, fill: '#9ca3af' }} tickLine={false} axisLine={false} interval="preserveStartEnd" />
              <YAxis hide />
              <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e5e7eb' }} itemStyle={{ color: chartColor(activeTab) }} formatter={(v) => [`${v}${s.unit}`, s.label]} labelStyle={{ color: '#6b7280' }} />
              <Line type="monotone" dataKey="v" stroke={chartColor(activeTab)} strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 현재 센서 수치 2×3 */}
      <div style={{ padding: '14px 16px 0' }}>
        <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 10 }}>현재 센서 수치</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          {Object.entries(sensors).map(([key, sv]) => (
            <div key={key} style={{ background: '#fff', borderRadius: 14, padding: 16, boxShadow: '0 1px 4px rgba(0,0,0,0.06)' }}>
              <div style={{ color: '#6b7280', fontSize: 12, marginBottom: 6 }}>{sv.label}</div>
              <div>
                <span style={{ fontSize: 26, fontWeight: 700, color: COLOR[sv.color]?.text }}>{sv.value}</span>
                <span style={{ fontSize: 13, color: '#9ca3af', marginLeft: 2 }}>{sv.unit}</span>
              </div>
              <div style={{ height: 4, background: '#f3f4f6', borderRadius: 99, margin: '10px 0 8px', overflow: 'hidden' }}>
                <div style={{ height: '100%', borderRadius: 99, background: COLOR[sv.color]?.bar, width: key === 'co2' ? `${Math.min(100, ((sv.value - 400) / 1600) * 100)}%` : key === 'light' ? `${Math.min(100, sv.value / 20)}%` : `${sv.value}%` }} />
              </div>
              <div style={{ color: '#9ca3af', fontSize: 11 }}>{sv.range}</div>
              <div style={{ color: COLOR[sv.color]?.text, fontSize: 13, fontWeight: 600, marginTop: 2 }}>{sv.status}</div>
            </div>
          ))}
        </div>
      </div>

      {/* 센서 기록 테이블 */}
      <div style={{ margin: '14px 16px 0', background: '#fff', borderRadius: 16, padding: 18, boxShadow: '0 1px 4px rgba(0,0,0,0.08)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
          <div style={{ fontWeight: 600, fontSize: 15 }}>센서 기록</div>
          <div style={{ color: '#9ca3af', fontSize: 12 }}>오늘 · 2분 간격</div>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ color: '#9ca3af', fontSize: 12 }}>
                {['시간','온도','습도','토양수분','CO₂','조도','수위'].map((h, hi) => (
                  <th key={h} style={{ textAlign: hi === 0 ? 'left' : 'right', paddingBottom: 10, fontWeight: 500, whiteSpace: 'nowrap', paddingRight: hi < 6 ? 12 : 0 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {logRows.map((row, i) => (
                <tr key={i} style={{ borderTop: '1px solid #f3f4f6' }}>
                  <td style={{ padding: '10px 12px 10px 0', color: '#6b7280' }}>{row.time}</td>
                  <td style={{ textAlign: 'right', padding: '10px 12px 10px 0' }}>{row.temp}</td>
                  <td style={{ textAlign: 'right', padding: '10px 12px 10px 0' }}>{row.hum}</td>
                  <td style={{ textAlign: 'right', padding: '10px 12px 10px 0', color: row.soilNum < 40 ? '#d97706' : '#1c1c1e', fontWeight: row.soilNum < 40 ? 600 : 400 }}>{row.soil}</td>
                  <td style={{ textAlign: 'right', padding: '10px 12px 10px 0', color: row.co2 > 1000 ? '#dc2626' : '#1c1c1e', fontWeight: row.co2 > 1000 ? 600 : 400 }}>{row.co2}</td>
                  <td style={{ textAlign: 'right', padding: '10px 12px 10px 0' }}>{row.light}</td>
                  <td style={{ textAlign: 'right', padding: '10px 0' }}>{row.water}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// ── Root ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [page, setPage]   = useState('home')
  const [data, setData]   = useState(() => transformRecords(MOCK_RECORDS))
  const [usingMock, setUsingMock] = useState(true)

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch(API_ENDPOINT)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const records = await res.json()
      const transformed = transformRecords(records)
      if (transformed) {
        setData(transformed)
        setUsingMock(false)
      }
    } catch {
      // 백엔드 미연결 시 mock 데이터 유지
    }
  }, [])

  useEffect(() => {
    fetchData()
    const id = setInterval(fetchData, POLL_INTERVAL_MS)
    return () => clearInterval(id)
  }, [fetchData])

  return (
    <div style={{ minHeight: '100vh', background: '#f2f2f7' }}>
      {/* 개발용 mock 표시 배너 (백엔드 연결 후 자동 사라짐) */}
      {usingMock && (
        <div style={{ background: '#fef3c7', borderBottom: '1px solid #fcd34d', padding: '6px 16px', textAlign: 'center', fontSize: 12, color: '#92400e' }}>
          📡 백엔드 미연결 — Mock 데이터 표시 중 ({API_ENDPOINT})
        </div>
      )}
      {page === 'home'
        ? <HomePage   data={data} onGoSensor={() => setPage('sensor')} />
        : <SensorPage data={data} onBack={() => setPage('home')} />
      }
    </div>
  )
}
