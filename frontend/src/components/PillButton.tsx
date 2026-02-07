type PillButtonProps = {
  label: string;
  active: boolean;
  onClick: () => void;
};

export default function PillButton({ label, active, onClick }: PillButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`rounded-full px-3 py-1 text-sm font-medium ${
        active ? "bg-slate-900 text-white" : "bg-slate-200 text-slate-700"
      }`}
    >
      {label}
    </button>
  );
}
