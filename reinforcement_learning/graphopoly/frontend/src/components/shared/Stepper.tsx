import React from 'react';
import { Button } from './Button';
import { Minus, Plus } from 'lucide-react';

interface StepperProps {
  value: number;
  onChange: (val: number) => void;
  min?: number;
  max?: number;
  step?: number;
  label?: string;
}

export const Stepper: React.FC<StepperProps> = ({ value, onChange, min = 0, max = Infinity, step = 1, label }) => {
  const handleDec = () => onChange(Math.max(min, value - step));
  const handleInc = () => onChange(Math.min(max, value + step));

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
      {label && <span className="text-body text-label">{label}</span>}
      <div style={{ display: 'flex', alignItems: 'center', gap: '4px', background: 'rgba(0,0,0,0.2)', padding: '4px', borderRadius: 'var(--radius-input)' }}>
        <Button variant="default" onClick={handleDec} disabled={value <= min} style={{ padding: '4px', border: 'none', background: 'transparent' }}>
          <Minus size={14} />
        </Button>
        <span className="text-body" style={{ minWidth: '24px', textAlign: 'center' }}>{value}</span>
        <Button variant="default" onClick={handleInc} disabled={value >= max} style={{ padding: '4px', border: 'none', background: 'transparent' }}>
          <Plus size={14} />
        </Button>
      </div>
    </div>
  );
};
