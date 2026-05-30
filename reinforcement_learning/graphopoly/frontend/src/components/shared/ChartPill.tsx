import React from 'react';

interface ChartPillProps {
  color: string;
  label: string;
  active: boolean;
  onClick: () => void;
}

export const ChartPill: React.FC<ChartPillProps> = ({ color, label, active, onClick }) => {
  return (
    <button
      onClick={onClick}
      style={{
        background: active ? `${color}25` : 'rgba(255,255,255,0.05)',
        border: `1px solid ${active ? color : 'rgba(255,255,255,0.1)'}`,
        color: active ? '#fff' : 'var(--color-text-dim)',
        borderRadius: '12px',
        padding: '2px 10px',
        fontSize: '11px',
        fontFamily: 'var(--font-mono)',
        cursor: 'pointer',
        transition: 'all 0.2s',
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        boxShadow: active ? `0 0 8px ${color}40` : 'none'
      }}
    >
      <div style={{ width: 8, height: 8, borderRadius: '50%', background: color }} />
      {label}
    </button>
  );
};
