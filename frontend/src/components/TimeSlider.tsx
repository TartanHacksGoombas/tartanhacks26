import { useCallback, useEffect, useMemo, useState } from "react";

type TimeSliderProps = {
  /** Currently selected day offset (0 = today, 1 = tomorrow, …, 6 = 6 days out) */
  value: number;
  /** Called when the user changes the slider */
  onChange: (dayOffset: number) => void;
  /** Max days out (default 7, matching NWS forecast range) */
  maxDays?: number;
};

/** Generate a label for a date offset. */
function dayLabel(offset: number): { main: string; sub: string } {
  if (offset === 0) return { main: "Today", sub: "Current conditions" };
  const d = new Date();
  d.setDate(d.getDate() + offset);
  const dayName = offset === 1 ? "Tomorrow" : d.toLocaleDateString(undefined, { weekday: "long" });
  const datePart = d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  return { main: dayName, sub: `${datePart} forecast` };
}

/** Short tick label for a day offset. */
function tickLabel(offset: number): string {
  if (offset === 0) return "Now";
  const d = new Date();
  d.setDate(d.getDate() + offset);
  return d.toLocaleDateString(undefined, { weekday: "short" });
}

export default function TimeSlider({ value, onChange, maxDays = 7 }: TimeSliderProps) {
  const days = useMemo(() => Array.from({ length: maxDays }, (_, i) => i), [maxDays]);

  // Keep a local value for smooth dragging, commit on release
  const [dragging, setDragging] = useState(false);
  const [localVal, setLocalVal] = useState(value);

  useEffect(() => { if (!dragging) setLocalVal(value); }, [value, dragging]);

  const handleInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setLocalVal(Number(e.target.value));
  }, []);

  const handleCommit = useCallback(() => {
    setDragging(false);
    onChange(localVal);
  }, [localVal, onChange]);

  const display = localVal;
  const { main, sub } = dayLabel(display);

  return (
    <div className="flex items-center gap-4 rounded-2xl border border-slate-200 bg-white/90 px-5 py-3 shadow-lg backdrop-blur">
      {/* Label */}
      <div className="min-w-[120px] shrink-0">
        <div className="text-base font-bold text-slate-900">{main}</div>
        <div className="text-xs text-slate-500">{sub}</div>
      </div>

      {/* Slider track */}
      <div className="relative flex-1 min-w-0">
        <input
          type="range"
          min={0}
          max={maxDays - 1}
          step={1}
          value={display}
          onChange={handleInput}
          onMouseDown={() => setDragging(true)}
          onMouseUp={handleCommit}
          onTouchStart={() => setDragging(true)}
          onTouchEnd={handleCommit}
          className="
            w-full cursor-pointer appearance-none rounded-full bg-slate-200 h-2.5
            accent-blue-600
            [&::-webkit-slider-thumb]:appearance-none
            [&::-webkit-slider-thumb]:h-6 [&::-webkit-slider-thumb]:w-6
            [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-600
            [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-white
            [&::-webkit-slider-thumb]:shadow-lg
            [&::-webkit-slider-thumb]:transition-transform [&::-webkit-slider-thumb]:hover:scale-110
          "
        />

        {/* Tick labels */}
        <div className="mt-1.5 flex justify-between px-[2px]">
          {days.map((d) => (
            <button
              key={d}
              onClick={() => onChange(d)}
              className={`text-[11px] leading-none transition-colors ${
                d === display ? "font-bold text-blue-700" : "text-slate-400 hover:text-slate-600"
              }`}
            >
              {tickLabel(d)}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
