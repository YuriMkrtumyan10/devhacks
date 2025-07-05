import React, { useState } from "react";
import LoginClean from "./LoginClean";
import Register from "./Register";

export default function AuthLayout() {
  const [isLoginMode, setIsLoginMode] = useState(true);

  return (
    <div className="auth-layout">
      <div className="auth-background">
        <div className="auth-bg-pattern"></div>
        <div className="auth-bg-gradient"></div>
      </div>

      <div className="auth-content">
        {isLoginMode ? (
          <LoginClean onSwitchToRegister={() => setIsLoginMode(false)} />
        ) : (
          <Register onSwitchToLogin={() => setIsLoginMode(true)} />
        )}
      </div>
    </div>
  );
}
