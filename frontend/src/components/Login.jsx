import React, { useState, useEffect } from "react";
import { useAuth } from "../context/AuthContext";

export default function Login({ onSwitchToRegister }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [validationErrors, setValidationErrors] = useState({});

  const { login, loading, error, clearError } = useAuth();

  useEffect(() => {
    clearError();
  }, []);

  const validateForm = () => {
    const errors = {};

    if (!email) {
      errors.email = "Email is required";
    } else if (!/\S+@\S+\.\S+/.test(email)) {
      errors.email = "Please enter a valid email";
    }

    if (!password) {
      errors.password = "Password is required";
    } else if (password.length < 6) {
      errors.password = "Password must be at least 6 characters";
    }

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) return;

    const result = await login(email, password);

    if (!result.success) {
      // Error is handled by the auth context
    }
  };

  const demoLogin = async (demoEmail) => {
    setEmail(demoEmail);
    setPassword("demo123");
    await login(demoEmail, "demo123");
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <div className="auth-logo">
            <div className="logo-icon">🏢</div>
            <h1>Welcome Back</h1>
            <p>Sign in to your account</p>
          </div>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="email">Email Address</label>
            <div className="input-wrapper">
              <input
                id="email"
                type="email"
                placeholder="Enter your email"
                value={email}
                onChange={(e) => {
                  setEmail(e.target.value);
                  if (validationErrors.email) {
                    setValidationErrors((prev) => ({ ...prev, email: "" }));
                  }
                }}
                className={validationErrors.email ? "error" : ""}
                disabled={loading}
              />
              <span className="input-icon">📧</span>
            </div>
            {validationErrors.email && (
              <span className="field-error">{validationErrors.email}</span>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <div className="input-wrapper">
              <input
                id="password"
                type={showPassword ? "text" : "password"}
                placeholder="Enter your password"
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  if (validationErrors.password) {
                    setValidationErrors((prev) => ({ ...prev, password: "" }));
                  }
                }}
                className={validationErrors.password ? "error" : ""}
                disabled={loading}
              />
              <button
                type="button"
                className="password-toggle"
                onClick={() => setShowPassword(!showPassword)}
                disabled={loading}
              >
                {showPassword ? "🙈" : "👁️"}
              </button>
            </div>
            {validationErrors.password && (
              <span className="field-error">{validationErrors.password}</span>
            )}
          </div>

          {error && (
            <div className="auth-error">
              <span className="error-icon">⚠️</span>
              {error}
            </div>
          )}

          <button type="submit" className="auth-submit-btn" disabled={loading}>
            {loading ? (
              <>
                <span className="loading-spinner"></span>
                Signing in...
              </>
            ) : (
              "Sign In"
            )}
          </button>
        </form>

        <div className="auth-divider">
          <span>or try demo accounts</span>
        </div>

        <div className="demo-accounts">
          <button
            type="button"
            className="demo-btn admin"
            onClick={() => demoLogin("admin@saas.com")}
            disabled={loading}
          >
            <span className="demo-icon">👑</span>
            <div>
              <div className="demo-title">Admin Demo</div>
              <div className="demo-desc">Full system access</div>
            </div>
          </button>

          <button
            type="button"
            className="demo-btn owner"
            onClick={() => demoLogin("owner@acme.com")}
            disabled={loading}
          >
            <span className="demo-icon">🏢</span>
            <div>
              <div className="demo-title">Company Owner</div>
              <div className="demo-desc">Acme Corp</div>
            </div>
          </button>

          <button
            type="button"
            className="demo-btn owner"
            onClick={() => demoLogin("owner@globex.com")}
            disabled={loading}
          >
            <span className="demo-icon">🌐</span>
            <div>
              <div className="demo-title">Company Owner</div>
              <div className="demo-desc">Globex Ltd</div>
            </div>
          </button>
        </div>

        <div className="auth-footer">
          <p>Don't have an account?</p>
          <button
            type="button"
            className="auth-switch-btn"
            onClick={onSwitchToRegister}
            disabled={loading || isSubmitting}
          >
            Create Account
          </button>
        </div>
      </div>
    </div>
  );
}
