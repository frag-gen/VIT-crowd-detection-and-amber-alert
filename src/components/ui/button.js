import React from 'react';

export const Button = ({ children, onClick, className = '', disabled = false }) => (
  <button
    className={`px-4 py-2 rounded font-semibold ${className} ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
    onClick={onClick}
    disabled={disabled}
  >
    {children}
  </button>
);

export default Button;
