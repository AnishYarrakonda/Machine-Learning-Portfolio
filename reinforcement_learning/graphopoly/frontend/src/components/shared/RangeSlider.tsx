import React from 'react';
import './shared.css';

interface RangeSliderProps {
  value: number;
  onChange: (val: number) => void;
  min: number;
  max: number;
  step?: number;
  label?: string;
  formatValue?: (val: number) => string;
}

export const RangeSlider: React.FC<RangeSliderProps> = ({ value, onChange, min, max, step = 1, label, formatValue }) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {(label || formatValue) && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          {label && <span className="text-body text-label">{label}</span>}
          {formatValue && <span className="text-body" style={{ color: 'var(--color-text-dim)', fontSize: '11px' }}>{formatValue(value)}</span>}
        </div>
      )}
      <input 
        type="range" 
        className="range-slider"
        min={min} 
        max={max} 
        step={step} 
        value={value} 
        onChange={(e) => onChange(parseFloat(e.target.value))} 
      />
    </div>
  );
};
