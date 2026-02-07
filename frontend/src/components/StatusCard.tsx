type StatusCardProps = {
  label: string;
  count: number;
  bgClass: string;
  textClass: string;
};

export default function StatusCard({ label, count, bgClass, textClass }: StatusCardProps) {
  return (
    <div className={`rounded-xl p-2 ${bgClass} ${textClass}`}>
      <div className="font-semibold">{label}</div>
      <div>{count}</div>
    </div>
  );
}
