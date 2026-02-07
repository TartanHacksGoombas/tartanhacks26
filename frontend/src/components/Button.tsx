type ButtonProps = {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  variant?: "primary" | "secondary";
  fullWidth?: boolean;
};

const styles = {
  primary:
    "rounded-lg bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-40",
  secondary:
    "rounded-lg bg-slate-200 px-3 py-2 text-sm text-slate-700 hover:bg-slate-300"
};

export default function Button({
  label,
  onClick,
  disabled = false,
  variant = "primary",
  fullWidth = false
}: ButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${styles[variant]} ${fullWidth ? "flex-1" : ""}`}
    >
      {label}
    </button>
  );
}
