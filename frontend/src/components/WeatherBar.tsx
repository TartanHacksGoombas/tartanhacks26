import { forwardRef, useCallback, useEffect, useState } from "react";
import { fetchWeather, fetchWeatherHourly, WeatherPeriod, HourlyPeriod } from "../utils/api";

/* ── Helpers ── */

function weatherIcon(forecast: string, isDaytime: boolean): string {
  const f = forecast.toLowerCase();
  if (f.includes("snow") || f.includes("blizzard")) return "❄️";
  if (f.includes("sleet") || f.includes("freezing")) return "🌨️";
  if (f.includes("thunder") || f.includes("storm")) return "⛈️";
  if (f.includes("rain") || f.includes("shower")) return "🌧️";
  if (f.includes("drizzle")) return "🌦️";
  if (f.includes("fog") || f.includes("mist") || f.includes("haze")) return "🌫️";
  if (f.includes("cloud") || f.includes("overcast")) return "☁️";
  if (f.includes("partly")) return isDaytime ? "⛅" : "🌙";
  if (f.includes("wind")) return "🌬️";
  if (f.includes("sunny") || f.includes("clear")) return isDaytime ? "☀️" : "🌙";
  return "🌤️";
}

function isHourDaytime(hour: number): boolean {
  return hour >= 6 && hour < 20;
}

function groupByDay(periods: WeatherPeriod[]): { day: WeatherPeriod; night: WeatherPeriod | null }[] {
  const groups: { day: WeatherPeriod; night: WeatherPeriod | null }[] = [];
  let i = 0;
  if (periods.length > 0 && !periods[0].isDaytime) {
    groups.push({ day: periods[0], night: null });
    i = 1;
  }
  while (i < periods.length) {
    const day = periods[i];
    const night = i + 1 < periods.length && !periods[i + 1].isDaytime ? periods[i + 1] : null;
    groups.push({ day, night });
    i += night ? 2 : 1;
  }
  return groups.slice(0, 7);
}

function getHoursForDay(hourly: HourlyPeriod[], dayStartTime: string): HourlyPeriod[] {
  const dayDate = new Date(dayStartTime).toDateString();
  return hourly.filter((h) => new Date(h.startTime).toDateString() === dayDate);
}

function formatHour(iso: string): string {
  const h = new Date(iso).getHours();
  return h === 0 ? "12a" : h < 12 ? `${h}a` : h === 12 ? "12p" : `${h - 12}p`;
}

/** Is the forecast wintry (snow/ice/sleet/freezing)? */
function isWintry(forecast: string): boolean {
  const f = forecast.toLowerCase();
  return f.includes("snow") || f.includes("ice") || f.includes("sleet") || f.includes("freezing") || f.includes("blizzard");
}

/** Color for precipitation — factors in type: rain stays yellow, snow/ice goes red. */
function precipColor(pct: number, forecast = ""): string {
  if (pct < 10) return "#16a34a";   // green — negligible
  if (pct < 20) return "#84cc16";   // lime

  if (isWintry(forecast)) {
    // Snow / ice / freezing — escalate to red fast
    if (pct >= 80) return "#7f1d1d"; // dark red
    if (pct >= 60) return "#b91c1c"; // red
    if (pct >= 40) return "#dc2626"; // bright red
    if (pct >= 20) return "#ea580c"; // orange
  }

  // Rain only — caps at yellow/orange
  if (pct >= 70) return "#ea580c";   // orange
  if (pct >= 50) return "#eab308";   // yellow
  if (pct >= 30) return "#facc15";   // light yellow
  return "#eab308";                   // yellow
}

/* ── Precipitation Graph ── */

const GRAPH_W = 460;
const GRAPH_H = 100;
const PAD_L = 30;
const PAD_R = 10;
const PAD_T = 10;
const PAD_B = 22;

function PrecipGraph({ hours }: { hours: HourlyPeriod[] }) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const plotW = GRAPH_W - PAD_L - PAD_R;
  const plotH = GRAPH_H - PAD_T - PAD_B;

  const n = hours.length;
  if (n === 0) return <div className="px-4 py-2 text-xs text-slate-400">No hourly data</div>;

  const barW = Math.max(2, plotW / n - 1);

  const toX = (i: number) => PAD_L + (i / n) * plotW + barW / 2;
  const toY = (pct: number) => PAD_T + plotH - (pct / 100) * plotH;

  const hovered = hoverIdx !== null ? hours[hoverIdx] : null;

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${GRAPH_W} ${GRAPH_H}`}
        className="w-full"
        style={{ maxHeight: 120 }}
        onMouseLeave={() => setHoverIdx(null)}
      >
        {/* Y-axis gridlines */}
        {[0, 25, 50, 75, 100].map((pct) => (
          <g key={pct}>
            <line
              x1={PAD_L} x2={GRAPH_W - PAD_R}
              y1={toY(pct)} y2={toY(pct)}
              stroke="#e2e8f0" strokeWidth={0.5}
            />
            <text x={PAD_L - 4} y={toY(pct) + 3} textAnchor="end" className="fill-slate-400" fontSize={8}>
              {pct}%
            </text>
          </g>
        ))}

        {/* Bars */}
        {hours.map((h, i) => {
          const pct = h.precipChance;
          const x = toX(i) - barW / 2;
          const barH = (pct / 100) * plotH;
          return (
            <rect
              key={h.startTime}
              x={x}
              y={PAD_T + plotH - barH}
              width={barW}
              height={Math.max(barH, 1)}
              rx={1}
              fill={precipColor(pct, h.shortForecast)}
              opacity={hoverIdx === i ? 1 : 0.8}
              onMouseEnter={() => setHoverIdx(i)}
              className="cursor-pointer"
            />
          );
        })}

        {/* Hover indicator line */}
        {hoverIdx !== null && (
          <line
            x1={toX(hoverIdx)} x2={toX(hoverIdx)}
            y1={PAD_T} y2={PAD_T + plotH}
            stroke="#475569" strokeWidth={0.8} strokeDasharray="2,2"
          />
        )}

        {/* X-axis labels (every 3 hours) */}
        {hours.map((h, i) => {
          const hr = new Date(h.startTime).getHours();
          if (hr % 3 !== 0) return null;
          return (
            <text
              key={h.startTime}
              x={toX(i)}
              y={GRAPH_H - 4}
              textAnchor="middle"
              className="fill-slate-400"
              fontSize={8}
            >
              {formatHour(h.startTime)}
            </text>
          );
        })}
      </svg>

      {/* Hover tooltip */}
      {hovered && hoverIdx !== null && (
        <div
          className="pointer-events-none absolute z-20 -translate-x-1/2 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs shadow-md"
          style={{
            left: `${(toX(hoverIdx) / GRAPH_W) * 100}%`,
            top: -4
          }}
        >
          <div className="font-semibold text-slate-700">
            {formatHour(hovered.startTime)} — {hovered.precipChance}% precip
          </div>
          <div className="text-slate-500">
            {weatherIcon(hovered.shortForecast, isHourDaytime(new Date(hovered.startTime).getHours()))}{" "}
            {hovered.shortForecast} · {hovered.temperature}°F
          </div>
          <div className="text-slate-400">
            Wind: {hovered.windSpeed} {hovered.windDirection}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Main Component ── */

type WeatherBarProps = {
  /** Day offset from the time slider (0 = today). When set, highlights that day. */
  activeDayOffset?: number;
};

const WeatherBar = forwardRef<HTMLDivElement, WeatherBarProps>(function WeatherBar({ activeDayOffset }, ref) {
  const [periods, setPeriods] = useState<WeatherPeriod[]>([]);
  const [hourly, setHourly] = useState<HourlyPeriod[]>([]);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  useEffect(() => {
    fetchWeather().then(setPeriods).catch(() => {});
    fetchWeatherHourly().then(setHourly).catch(() => {});
  }, []);

  const days = groupByDay(periods);

  // Sync expanded panel with time slider's day offset
  useEffect(() => {
    if (activeDayOffset !== undefined && days.length > 0) {
      if (activeDayOffset < days.length) {
        setSelectedIdx(activeDayOffset);
      }
    }
  }, [activeDayOffset, days.length]);

  if (days.length === 0) return null;

  const selected = selectedIdx !== null ? days[selectedIdx] : null;
  const selectedHours = selected ? getHoursForDay(hourly, selected.day.startTime) : [];

  return (
    <div ref={ref} className="absolute right-4 top-4 z-10 w-[500px] rounded-2xl border border-slate-200 bg-white/90 shadow-lg backdrop-blur">
      {/* Day row */}
      <div className="flex justify-between px-2 pt-2 pb-1">
        {days.map((g, i) => {
          const isActive = selectedIdx === i;
          const dayHours = getHoursForDay(hourly, g.day.startTime);
          const maxPrecip = dayHours.length > 0 ? Math.max(...dayHours.map((h) => h.precipChance)) : 0;
          const worstForecast = dayHours.length > 0
            ? dayHours.reduce((worst, h) => h.precipChance > worst.precipChance ? h : worst).shortForecast
            : "";
          return (
            <button
              key={g.day.name}
              onClick={() => setSelectedIdx(isActive ? null : i)}
              className={`flex flex-col items-center gap-0.5 rounded-xl px-2.5 py-1.5 transition-colors ${
                isActive ? "bg-blue-100" : "hover:bg-slate-100"
              }`}
            >
              <span className="text-[11px] font-medium text-slate-500">
                {g.day.name.slice(0, 3)}
              </span>
              <span className="text-[9px] text-slate-400">
                {new Date(g.day.startTime).toLocaleDateString(undefined, { month: "numeric", day: "numeric" })}
              </span>
              <span className="text-xl leading-none">
                {weatherIcon(g.day.shortForecast, g.day.isDaytime)}
              </span>
              <span className="text-xs font-semibold text-slate-800">
                {g.day.temperature}°
              </span>
              <span className="text-[9px] font-medium" style={{ color: precipColor(maxPrecip, worstForecast) }}>
                {maxPrecip}%
              </span>
            </button>
          );
        })}
      </div>

      {/* Expanded detail panel */}
      {selected && (
        <div className="border-t border-slate-200 px-3 py-3 text-sm">
          <div className="mb-1 flex items-baseline justify-between">
            <div>
              <span className="font-semibold text-slate-800">{selected.day.name}</span>
              <span className="ml-2 text-xs text-slate-400">
                {new Date(selected.day.startTime).toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" })}
              </span>
            </div>
            <div className="flex gap-3 text-xs text-slate-500">
              <span>
                {weatherIcon(selected.day.shortForecast, true)} {selected.day.temperature}°F {selected.day.shortForecast}
              </span>
              {selected.night && (
                <span>
                  {weatherIcon(selected.night.shortForecast, false)} {selected.night.temperature}°F
                </span>
              )}
            </div>
          </div>

          {/* Precipitation graph */}
          <div className="mt-1">
            <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-slate-400">
              Precipitation Chance
            </div>
            <PrecipGraph hours={selectedHours} />
          </div>
        </div>
      )}
    </div>
  );
});

export default WeatherBar;
