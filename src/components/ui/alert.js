import React from 'react';

export const Alert = ({ children, variant = 'info' }) => {
  const variantClasses = {
    info: 'bg-blue-100 text-blue-700',
    warning: 'bg-yellow-100 text-yellow-700',
    destructive: 'bg-red-100 text-red-700',
  };

  return (
    <div className={`p-4 rounded-md border ${variantClasses[variant]}`}>
      {children}
    </div>
  );
};

export const AlertTitle = ({ children }) => (
  <h3 className="font-bold text-lg">{children}</h3>
);

export const AlertDescription = ({ children }) => (
  <p className="text-sm">{children}</p>
);

export default Alert;
