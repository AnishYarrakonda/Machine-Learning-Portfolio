import React from 'react';
import './shared.css';

interface NumberInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  value: number | '';
  onChangeValue: (val: number | '') => void;
}

export const NumberInput: React.FC<NumberInputProps> = ({ label, value, onChangeValue, className = '', style, ...props }) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', width: '100%', ...style }}>
      {label && <label className="text-body text-label">{label}</label>}
      <input 
        type="number" 
        className={`input-base ${className}`} 
        value={value}
        onChange={(e) => {
          const val = e.target.value;
          onChangeValue(val === '' ? '' : parseFloat(val));
        }}
        {...props}
      />
    </div>
  );
};
