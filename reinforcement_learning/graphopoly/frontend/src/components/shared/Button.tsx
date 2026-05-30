import React, { forwardRef } from 'react';
import './shared.css';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'primary' | 'danger' | 'warning' | 'ghost' | 'secondary';
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(({ children, variant = 'default', className = '', ...props }, ref) => {
  return (
    <button
      ref={ref}
      className={`btn ${variant !== 'default' ? variant : ''} ${className}`}
      style={{
        ...props.style
      }}
      {...props}
    >
      {children}
    </button>
  );
});

Button.displayName = 'Button';
