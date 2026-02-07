import { useCallback, useRef, useState } from "react";
import { geocode, resolvePlace, GeocodeSuggestion } from "../api";

type SearchInputProps = {
  value: string;
  onChange: (text: string) => void;
  onSelect: (suggestion: GeocodeSuggestion) => void;
  placeholder: string;
  /** Whether a coordinate has already been selected */
  hasCoord: boolean;
};

export default function SearchInput({ value, onChange, onSelect, placeholder, hasCoord }: SearchInputProps) {
  const [suggestions, setSuggestions] = useState<GeocodeSuggestion[]>([]);
  const [loading, setLoading] = useState(false);
  const [focused, setFocused] = useState(false);
  const [resolving, setResolving] = useState(false);
  const seqRef = useRef(0);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const blurTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleChange = useCallback(
    (text: string) => {
      onChange(text);
      if (debounceRef.current) clearTimeout(debounceRef.current);
      if (text.length < 3) {
        setSuggestions([]);
        setLoading(false);
        return;
      }
      setLoading(true);
      debounceRef.current = setTimeout(() => {
        const seq = ++seqRef.current;
        geocode(text)
          .then((results) => {
            if (seq === seqRef.current) setSuggestions(results);
          })
          .catch(() => {
            if (seq === seqRef.current) setSuggestions([]);
          })
          .finally(() => {
            if (seq === seqRef.current) setLoading(false);
          });
      }, 600);
    },
    [onChange]
  );

  const handlePick = useCallback(
    async (s: GeocodeSuggestion) => {
      setSuggestions([]);
      setLoading(false);
      setResolving(true);
      try {
        const coords = await resolvePlace(s.placeId);
        if (coords) {
          onSelect({ ...s, lat: coords.lat, lng: coords.lng });
        }
      } finally {
        setResolving(false);
      }
    },
    [onSelect]
  );

  const handleFocus = useCallback(() => {
    if (blurTimeoutRef.current) clearTimeout(blurTimeoutRef.current);
    setFocused(true);
  }, []);

  const handleBlur = useCallback(() => {
    // Short delay so clicking a suggestion registers before we hide the dropdown
    blurTimeoutRef.current = setTimeout(() => {
      setFocused(false);
      setSuggestions([]);
      setLoading(false);
    }, 200);
  }, []);

  const showDropdown = focused && !resolving;

  return (
    <div className="relative">
      <input
        value={value}
        onChange={(e) => handleChange(e.target.value)}
        onFocus={handleFocus}
        onBlur={handleBlur}
        placeholder={placeholder}
        disabled={resolving}
        className={`w-full rounded-lg border px-3 py-2 pr-8 text-sm focus:outline-none disabled:opacity-60 ${
          hasCoord
            ? "border-green-400 bg-green-50/50 focus:border-green-500"
            : "border-slate-300 focus:border-blue-500"
        }`}
      />
      {hasCoord && !resolving && (
        <div className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-green-500">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
            <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 0 1 .143 1.052l-8 10.5a.75.75 0 0 1-1.127.075l-4.5-4.5a.75.75 0 0 1 1.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 0 1 1.05-.143Z" clipRule="evenodd" />
          </svg>
        </div>
      )}
      {resolving && (
        <div className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-slate-400">Loading…</div>
      )}
      {showDropdown && loading && suggestions.length === 0 && value.length >= 3 && (
        <div className="absolute left-0 right-0 top-full z-30 mt-1 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-400 shadow-lg">
          Searching...
        </div>
      )}
      {showDropdown && !loading && suggestions.length === 0 && value.length >= 3 && !hasCoord && (
        <div className="absolute left-0 right-0 top-full z-30 mt-1 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-400 shadow-lg">
          No results — try adding a street name
        </div>
      )}
      {showDropdown && suggestions.length > 0 && (
        <ul className="absolute left-0 right-0 top-full z-30 mt-1 max-h-40 overflow-auto rounded-lg border border-slate-200 bg-white shadow-lg">
          {suggestions.map((s, i) => (
            <li
              key={i}
              onMouseDown={(e) => e.preventDefault()} // Prevent blur before click registers
              onClick={() => handlePick(s)}
              className="cursor-pointer px-3 py-2 text-xs hover:bg-blue-50 truncate"
            >
              {s.displayName}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
