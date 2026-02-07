type ToggleButtonProps = {
  open: boolean;
  openLabel: string;
  closedLabel: string;
  onClick: () => void;
};

export default function ToggleButton({ open, openLabel, closedLabel, onClick }: ToggleButtonProps) {
  return (
    <button
      onClick={onClick}
      className="flex w-full items-center justify-between rounded-lg bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700"
    >
      <span>{open ? openLabel : closedLabel}</span>
      <span className="text-lg leading-none">{open ? "\u00D7" : "\u2192"}</span>
    </button>
  );
}
