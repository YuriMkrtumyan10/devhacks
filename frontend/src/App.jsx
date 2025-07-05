import React from "react";
import {
  Link,
  Navigate,
  Route,
  BrowserRouter as Router,
  Routes,
  useLocation,
} from "react-router-dom";

import Chat from "./components/Chat";
import Company from "./components/CompanyOptimized";
import CompanySettings from "./components/CompanySettings";
import AdminDashboard from "./components/AdminDashboard";
import OwnerDashboard from "./components/OwnerDashboard";
import AuthLayout from "./components/AuthLayout";
import Products from "./components/Products";
import Navigation from "./components/Navigation";
import { TenantProvider } from "./components/TenantContext";
import TenantSwitcher from "./components/TenantSwitcher";
import TenantAnalytics from "./components/TenantAnalytics";
import { AuthProvider, useAuth } from "./context/AuthContext";

function DashboardRouter() {
  const { user } = useAuth();

  if (user?.role === "admin") {
    return <AdminDashboard />;
  }

  return <OwnerDashboard />;
}

function AppContent() {
  const { user, logout, loading } = useAuth();

  // Only show tenant switcher for admin
  const isAdmin = user?.role === "admin";

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading...</p>
      </div>
    );
  }

  if (!user) {
    return <AuthLayout />;
  }

  return (
    <TenantProvider>
      <Router>
        <header className="modern-header">
          <div className="header-container">
            <div className="header-brand">
              <div className="brand-logo">🚀</div>
              <div className="brand-info">
                <h1>{user?.role === "admin" ? "Admin Panel" : "Dashboard"}</h1>
                <p>
                  {user?.role === "admin"
                    ? "Manage all companies"
                    : "Your business insights"}
                </p>
              </div>
            </div>

            <Navigation />

            <div className="header-actions">
              {isAdmin && <TenantSwitcher />}
              <div className="user-menu">
                <div className="user-avatar">
                  {user.email.charAt(0).toUpperCase()}
                </div>
                <div className="user-details">
                  <span className="user-name">{user.email}</span>
                  <span className="user-role">{user.role}</span>
                </div>
                <button className="logout-btn" onClick={logout}>
                  Logout
                </button>
              </div>
            </div>
          </div>
        </header>
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<DashboardRouter />} />
            <Route path="/companies" element={<Company user={user} />} />
            <Route
              path="/company"
              element={<CompanySettings companyId={user.companyId} />}
            />
            <Route path="/products" element={<Products />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/analytics" element={<TenantAnalytics />} />
            <Route
              path="/users"
              element={<div>Users Management (Coming Soon)</div>}
            />
            <Route
              path="/orders"
              element={<div>Orders Management (Coming Soon)</div>}
            />
            <Route
              path="/team"
              element={<div>Team Management (Coming Soon)</div>}
            />
          </Routes>
        </main>
      </Router>
    </TenantProvider>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
