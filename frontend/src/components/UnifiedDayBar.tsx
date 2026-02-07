import { forwardRef, useCallback, useEffect, useRef, useState } from "react";
import { fetchWeather, fetchWeatherHourly, extractWeatherParams, WeatherPeriod, HourlyPeriod } from "../utils/api";
import { WeatherParams } from "../types";

/* ── Helpers (migrated from WeatherBar) ── */

function weatherIcon(forecast: string, isDaytime: boolean): string {
  const f = forecast.toLowerCase();
  if (f.includes("snow") || f.includes("blizzard")) return "\u2744\uFE0F";
  if (f.includes("sleet") || f.includes("freezing")) return "\uD83C\uDF28\uFE0F";
  if (f.includes("thunder") || f.includes("storm")) return "\u26C8\uFE0F";
  if (f.includes("rain") || f.includes("shower")) return "\uD83C\uDF27\uFE0F";
  if (f.includes("drizzle")) return "\uD83C\uDF26\uFE0F";
  if (f.includes("fog") || f.includes("mist") || f.includes("haze")) return "\uD83C\uDF2B\uFE0F";
  if (f.includes("cloud") || f.includes("overcast")) return "\u2601\uFE0F";
  if (f.includes("partly")) return isDaytime ? "\u26C5" : "\uD83C\uDF19";
  if (f.includes("wind")) return "\uD83C\uDF2C\uFE0F";
  if (f.includes("sunny") || f.includes("clear")) return isDaytime ? "\u2600\uFE0F" : "\uD83C\uDF19";
  return "\uD83C\uDF24\uFE0F";
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

function isWintry(forecast: string): boolean {
  const f = forecast.toLowerCase();
  return f.includes("snow") || f.includes("ice") || f.includes("sleet") || f.includes("freezing") || f.includes("blizzard");
}

function precipColor(pct: number, forecast = ""): string {
  if (pct < 10) return "#16a34a";
  if (pct < 20) return "#84cc16";
  if (isWintry(forecast)) {
    if (pct >= 80) return "#7f1d1d";
    if (pct >= 60) return "#b91c1c";
    if (pct >= 40) return "#dc2626";
    if (pct >= 20) return "#ea580c";
  }
  if (pct >= 70) return "#ea580c";
  if (pct >= 50) return "#eab308";
  if (pct >= 30) return "#facc15";
  return "#eab308";
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
        {[0, 25, 50, 75, 100].map((pct) => (
          <g key={pct}>
            <line x1={PAD_L} x2={GRAPH_W - PAD_R} y1={toY(pct)} y2={toY(pct)} stroke="#e2e8f0" strokeWidth={0.5} />
            <text x={PAD_L - 4} y={toY(pct) + 3} textAnchor="end" className="fill-slate-400" fontSize={8}>{pct}%</text>
          </g>
        ))}
        {hours.map((h, i) => {
          const pct = h.precipChance;
          const x = toX(i) - barW / 2;
          const barH = (pct / 100) * plotH;
          return (
            <rect
              key={h.startTime}
              x={x} y={PAD_T + plotH - barH}
              width={barW} height={Math.max(barH, 1)}
              rx={1}
              fill={precipColor(pct, h.shortForecast)}
              opacity={hoverIdx === i ? 1 : 0.8}
              onMouseEnter={() => setHoverIdx(i)}
              className="cursor-pointer"
            />
          );
        })}
        {hoverIdx !== null && (
          <line x1={toX(hoverIdx)} x2={toX(hoverIdx)} y1={PAD_T} y2={PAD_T + plotH} stroke="#475569" strokeWidth={0.8} strokeDasharray="2,2" />
        )}
        {hours.map((h, i) => {
          const hr = new Date(h.startTime).getHours();
          if (hr % 3 !== 0) return null;
          return (
            <text key={h.startTime} x={toX(i)} y={GRAPH_H - 4} textAnchor="middle" className="fill-slate-400" fontSize={8}>
              {formatHour(h.startTime)}
            </text>
          );
        })}
      </svg>
      {hovered && hoverIdx !== null && (
        <div
          className="pointer-events-none absolute z-20 -translate-x-1/2 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs shadow-md"
          style={{ left: `${(toX(hoverIdx) / GRAPH_W) * 100}%`, top: -4 }}
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

type UnifiedDayBarProps = {
  value: number;
  onChange: (dayOffset: number) => void;
  onSnowDetected?: (params: WeatherParams) => void;
};

const UnifiedDayBar = forwardRef<HTMLDivElement, UnifiedDayBarProps>(function UnifiedDayBar(
  { value, onChange, onSnowDetected },
  ref
) {
  const [periods, setPeriods] = useState<WeatherPeriod[]>([]);
  const [hourly, setHourly] = useState<HourlyPeriod[]>([]);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const snowDetectedRef = useRef(false);

  // Fetch weather data
  useEffect(() => {
    fetchWeather().then(setPeriods).catch(() => {});
    fetchWeatherHourly().then(setHourly).catch(() => {});
  }, []);

  // Auto-detect snow
  useEffect(() => {
    if (snowDetectedRef.current || periods.length === 0 || hourly.length === 0) return;
    if (!onSnowDetected) return;
    const params = extractWeatherParams(periods, hourly);
    if (params) {
      snowDetectedRef.current = true;
      onSnowDetected(params);
    }
  }, [periods, hourly, onSnowDetected]);

  const days = groupByDay(periods);

  const handleDayClick = useCallback((idx: number) => {
    if (idx === value) {
      // Toggle expand on already-selected day
      setExpandedIdx((prev) => (prev === idx ? null : idx));
    } else {
      onChange(idx);
      setExpandedIdx(null);
    }
  }, [value, onChange]);

  const expanded = expandedIdx !== null ? days[expandedIdx] : null;
  const expandedHours = expanded ? getHoursForDay(hourly, expanded.day.startTime) : [];

  if (days.length === 0) {
    return (
      <div ref={ref} className="rounded-2xl border border-slate-200 bg-white/90 px-5 py-3 shadow-lg backdrop-blur">
        <div className="text-sm text-slate-400">Loading weather...</div>
      </div>
    );
  }

  return (
    <div ref={ref} className="rounded-2xl border border-slate-200 bg-white/90 shadow-lg backdrop-blur">
      {/* Day columns */}
      <div className="flex justify-between px-2 pt-2 pb-2">
        {days.map((g, i) => {
          const isSelected = i === value;
          const dayHours = getHoursForDay(hourly, g.day.startTime);
          const maxPrecip = dayHours.length > 0 ? Math.max(...dayHours.map((h) => h.precipChance)) : 0;
          const worstForecast = dayHours.length > 0
            ? dayHours.reduce((worst, h) => h.precipChance > worst.precipChance ? h : worst).shortForecast
            : "";
          return (
            <button
              key={g.day.name}
              onClick={() => handleDayClick(i)}
              className={`flex flex-col items-center gap-0.5 rounded-xl px-2.5 py-1.5 transition-all ${
                isSelected
                  ? "bg-blue-100 ring-2 ring-blue-400"
                  : "hover:bg-slate-100"
              }`}
            >
              <span className="text-[11px] font-medium text-slate-500">
                {i === 0 ? "Today" : g.day.name.slice(0, 3)}
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

      {/* Expandable hourly detail */}
      {expanded && (
        <div className="border-t border-slate-200 px-3 py-3 text-sm">
          <div className="mb-1 flex items-baseline justify-between">
            <div>
              <span className="font-semibold text-slate-800">{expanded.day.name}</span>
              <span className="ml-2 text-xs text-slate-400">
                {new Date(expanded.day.startTime).toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" })}
              </span>
            </div>
            <div className="flex gap-3 text-xs text-slate-500">
              <span>
                {weatherIcon(expanded.day.shortForecast, true)} {expanded.day.temperature}°F {expanded.day.shortForecast}
              </span>
              {expanded.night && (
                <span>
                  {weatherIcon(expanded.night.shortForecast, false)} {expanded.night.temperature}°F
                </span>
              )}
            </div>
          </div>
          <div className="mt-1">
            <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-slate-400">
              Precipitation Chance
            </div>
            <PrecipGraph hours={expandedHours} />
          </div>
        </div>
      )}
    </div>
  );
});

export default UnifiedDayBar;
