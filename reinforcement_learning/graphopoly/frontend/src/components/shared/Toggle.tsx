import React from 'react';
import './shared.css';

interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  disabled?: boolean;
}

export const Toggle: React.FC<ToggleProps> = ({ checked, onChange, label, disabled }) => {
  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation(); // prevent double-fire if parent also has onClick
    if (disabled) return;
    onChange(!checked);
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <div
        className={`toggle-track ${checked ? 'active' : ''} ${disabled ? 'disabled' : ''}`}
        onClick={handleClick}
        style={{ cursor: disabled ? 'not-allowed' : 'pointer' }}
      >
        <div className="toggle-thumb" />
      </div>
      {label && (
        <span
          className="text-body text-label"
          onClick={handleClick}
          style={{ cursor: disabled ? 'not-allowed' : 'pointer' }}
        >
          {label}
        </span>
      )}
    </div>
  );
};
