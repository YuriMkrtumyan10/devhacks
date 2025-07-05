import React from "react";
import { Link, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const navigationConfig = {
  admin: [
    { path: "/dashboard", label: "Dashboard", icon: "📊" },
    { path: "/companies", label: "Companies", icon: "🏢" },
    { path: "/analytics", label: "Analytics", icon: "📈" },
    { path: "/users", label: "Users", icon: "👥" },
  ],
  owner: [
    { path: "/dashboard", label: "Dashboard", icon: "📊" },
    { path: "/company", label: "Company Settings", icon: "⚙️" },
    { path: "/products", label: "Products", icon: "📦" },
    { path: "/orders", label: "Orders", icon: "🛍️" },
    { path: "/team", label: "Team", icon: "👥" },
  ],
  user: [
    { path: "/dashboard", label: "Dashboard", icon: "📊" },
    { path: "/company", label: "Company Settings", icon: "⚙️" },
    { path: "/products", label: "Products", icon: "📦" },
  ],
};

export default function Navigation() {
  const { pathname } = useLocation();
  const { user } = useAuth();

  const navItems = navigationConfig[user?.role] || navigationConfig.user;

  return (
    <nav className="modern-navigation">
      <div className="nav-container">
        {navItems.map((item) => (
          <Link
            key={item.path}
            to={item.path}
            className={`nav-item text-sm md:text-base ${pathname === item.path ? "active" : ""}`}
          >
            <span className="nav-icon text-base">{item.icon}</span>
            <span className="nav-label hidden sm:inline">{item.label}</span>
            {pathname === item.path && <div className="nav-indicator" />}
          </Link>
        ))}
      </div>
    </nav>
  );
}
