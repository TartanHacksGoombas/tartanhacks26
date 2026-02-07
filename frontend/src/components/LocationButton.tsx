import { useState } from "react";

type LocationState = "idle" | "loading" | "active";

interface LocationButtonProps {
  onLocation: (lng: number, lat: number, accuracy: number) => void;
}

export default function LocationButton({ onLocation }: LocationButtonProps) {
  const [state, setState] = useState<LocationState>("idle");

  const handleClick = () => {
    if (!navigator.geolocation) {
      alert("Geolocation is not supported by your browser.");
      return;
    }

    setState("loading");

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setState("active");
        onLocation(pos.coords.longitude, pos.coords.latitude, pos.coords.accuracy);
      },
      (err) => {
        console.error("Geolocation error:", err);
        setState("idle");
        if (err.code === err.PERMISSION_DENIED) {
          alert("Location access was denied. Please enable it in your browser settings.");
        }
      },
      { enableHighAccuracy: true, timeout: 10000 }
    );
  };

  return (
    <button
      onClick={handleClick}
      title="Show my location"
      className={`flex items-center justify-center w-[30px] h-[30px] rounded shadow border transition-colors ${
        state === "active"
          ? "bg-blue-50 border-blue-400"
          : "bg-white border-slate-300 hover:bg-slate-50"
      }`}
    >
      {state === "loading" ? (
        <svg className="w-4 h-4 text-blue-500 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <circle cx="12" cy="12" r="10" strokeDasharray="31.4 31.4" strokeLinecap="round" />
        </svg>
      ) : (
        <svg
          className={`w-4 h-4 ${state === "active" ? "text-blue-500" : "text-slate-600"}`}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          {/* Crosshair icon */}
          <circle cx="12" cy="12" r="4" />
          <line x1="12" y1="2" x2="12" y2="6" />
          <line x1="12" y1="18" x2="12" y2="22" />
          <line x1="2" y1="12" x2="6" y2="12" />
          <line x1="18" y1="12" x2="22" y2="12" />
        </svg>
      )}
    </button>
  );
}
