import React, { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
} from "recharts";
import { dashboardApi } from "../services/api";
import { useTenant } from "./TenantContext";
import { useAuth } from "../context/AuthContext";

const COLORS = [
  "#5f6fff",
  "#27ae60",
  "#e67e22",
  "#e74c3c",
  "#aab6ff",
  "#3b3bff",
  "#f39c12",
  "#9b59b6",
];

export default function TenantAnalytics() {
  const { tenant, tenantId, isAdmin } = useTenant();
  const { user } = useAuth();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeFilter, setTimeFilter] = useState("30d");

  useEffect(() => {
    async function fetchAnalytics() {
      if (!tenantId && !user?.companyId) return;

      setLoading(true);
      setError(null);

      try {
        let analyticsData;

        if (isAdmin && tenantId) {
          // Admin viewing specific company analytics
          analyticsData = await dashboardApi.getCompanyStats(
            tenantId,
            timeFilter,
          );
        } else if (user?.companyId) {
          // Regular user viewing their company analytics
          analyticsData = await dashboardApi.getCompanyStats(
            user.companyId,
            timeFilter,
          );
        } else {
          // Fallback to admin dashboard if no specific company selected
          analyticsData = await dashboardApi.getAdminStats(timeFilter);
        }

        setData(analyticsData);
      } catch (err) {
        setError(err.message);
        console.error("Failed to fetch analytics:", err);
      } finally {
        setLoading(false);
      }
    }

    fetchAnalytics();
  }, [tenantId, user?.companyId, isAdmin, timeFilter]);

  if (loading) {
    return (
      <div className="modern-container">
        <div className="loading-state">
          <div className="loading-spinner"></div>
          <p>Loading analytics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="modern-container">
        <div className="error-state">
          <div className="error-icon">⚠️</div>
          <h3>Failed to load analytics</h3>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="modern-container">
        <div className="empty-state">
          <div className="empty-icon">📊</div>
          <h3>No analytics data available</h3>
          <p>
            {isAdmin
              ? "Select a company from the dropdown to view analytics"
              : "No data available for your company"}
          </p>
        </div>
      </div>
    );
  }

  const currentCompanyName =
    isAdmin && tenant ? tenant.name : user?.company?.name || "Company";

  return (
    <div className="modern-container">
      <div className="page-header">
        <div className="header-main">
          <div className="header-icon">📊</div>
          <div className="header-text">
            <h1>Analytics Dashboard</h1>
            <div className="company-context">
              <span className="company-name">{currentCompanyName}</span>
              {isAdmin && tenant && (
                <div className="company-details">
                  <span className="industry-tag">{tenant.industry}</span>
                  <span className="country-tag">{tenant.country}</span>
                  {tenant.founded_year && (
                    <span className="year-tag">Est. {tenant.founded_year}</span>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="header-actions">
          <div className="filter-container">
            <label htmlFor="timeFilter">Period:</label>
            <select
              id="timeFilter"
              value={timeFilter}
              onChange={(e) => setTimeFilter(e.target.value)}
              className="modern-select"
            >
              <option value="7d">Last 7 days</option>
              <option value="30d">Last 30 days</option>
              <option value="90d">Last 90 days</option>
              <option value="1y">Last year</option>
            </select>
          </div>
        </div>
      </div>

      {/* Key Performance Metrics */}
      <div className="analytics-section">
        <h2 className="section-title">Key Performance Metrics</h2>
        <div className="stats-grid">
          <div className="stat-card products">
            <div className="stat-header">
              <div className="stat-icon">📦</div>
              <span className="stat-label">Products</span>
            </div>
            <div className="stat-value">{data.totalProducts || 0}</div>
            <div className="stat-change positive">
              <span className="change-icon">↗</span>+{data.newProducts || 0}{" "}
              this period
            </div>
          </div>

          <div className="stat-card revenue">
            <div className="stat-header">
              <div className="stat-icon">💰</div>
              <span className="stat-label">Revenue</span>
            </div>
            <div className="stat-value">
              ${(data.revenue || 0).toLocaleString()}
            </div>
            <div
              className={`stat-change ${data.revenueGrowth >= 0 ? "positive" : "negative"}`}
            >
              <span className="change-icon">
                {data.revenueGrowth >= 0 ? "↗" : "↘"}
              </span>
              {data.revenueGrowth >= 0 ? "+" : ""}
              {data.revenueGrowth || 0}% growth
            </div>
          </div>

          <div className="stat-card orders">
            <div className="stat-header">
              <div className="stat-icon">📋</div>
              <span className="stat-label">Orders</span>
            </div>
            <div className="stat-value">{data.orders || 0}</div>
            <div className="stat-change positive">
              <span className="change-icon">↗</span>+{data.newOrders || 0} new
              orders
            </div>
          </div>

          <div className="stat-card customers">
            <div className="stat-header">
              <div className="stat-icon">👥</div>
              <span className="stat-label">Customers</span>
            </div>
            <div className="stat-value">{data.customers || 0}</div>
            <div className="stat-change positive">
              <span className="change-icon">↗</span>+{data.newCustomers || 0}{" "}
              new customers
            </div>
          </div>
        </div>
      </div>

      {/* Revenue Trends */}
      <div className="analytics-section">
        <div className="chart-card">
          <div className="chart-header">
            <h3 className="chart-title">Revenue & Orders Trend</h3>
            <div className="chart-legend">
              <div className="legend-item">
                <div className="legend-dot revenue"></div>
                <span>Revenue</span>
              </div>
              <div className="legend-item">
                <div className="legend-dot orders"></div>
                <span>Orders</span>
              </div>
            </div>
          </div>
          <div className="chart-wrapper">
            {data.revenueChart && data.revenueChart.length > 0 ? (
              <ResponsiveContainer width="100%" height={350}>
                <LineChart
                  data={data.revenueChart}
                  margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                >
                  <XAxis
                    dataKey="month"
                    tick={{ fontSize: 12 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fontSize: 12 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "rgba(255, 255, 255, 0.95)",
                      border: "none",
                      borderRadius: "12px",
                      boxShadow: "0 8px 32px rgba(0, 0, 0, 0.1)",
                    }}
                    formatter={(value, name) => [
                      name === "revenue" ? `$${value.toLocaleString()}` : value,
                      name === "revenue" ? "Revenue" : "Orders",
                    ]}
                  />
                  <Line
                    type="monotone"
                    dataKey="revenue"
                    stroke="#6366f1"
                    strokeWidth={3}
                    dot={{ fill: "#6366f1", strokeWidth: 2, r: 5 }}
                    activeDot={{ r: 7, fill: "#6366f1" }}
                  />
                  <Line
                    type="monotone"
                    dataKey="orders"
                    stroke="#10b981"
                    strokeWidth={3}
                    dot={{ fill: "#10b981", strokeWidth: 2, r: 5 }}
                    activeDot={{ r: 7, fill: "#10b981" }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="chart-empty">
                <div className="empty-icon">📈</div>
                <p>No revenue data available</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Product Performance */}
      {data.productPerformance && data.productPerformance.length > 0 && (
        <div className="analytics-section">
          <div className="chart-card">
            <div className="chart-header">
              <h3 className="chart-title">Product Performance</h3>
              <p className="chart-subtitle">Revenue by product category</p>
            </div>
            <div className="chart-wrapper">
              <ResponsiveContainer width="100%" height={320}>
                <BarChart
                  data={data.productPerformance}
                  margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                >
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 12 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fontSize: 12 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "rgba(255, 255, 255, 0.95)",
                      border: "none",
                      borderRadius: "12px",
                      boxShadow: "0 8px 32px rgba(0, 0, 0, 0.1)",
                    }}
                    formatter={(value) => [
                      `$${value.toLocaleString()}`,
                      "Revenue",
                    ]}
                  />
                  <Bar dataKey="revenue" fill="#6366f1" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Customer Insights */}
      {data.customerMetrics && (
        <div className="analytics-section">
          <h2 className="section-title">Customer Insights</h2>
          <div className="customer-metrics-grid">
            <div className="customer-metric-card retention">
              <div className="metric-header">
                <div className="metric-icon">🔄</div>
                <span className="metric-label">Retention Rate</span>
              </div>
              <div className="metric-value">
                {data.customerMetrics.retention}%
              </div>
              <div className="metric-description">
                Customer retention this period
              </div>
            </div>

            <div className="customer-metric-card satisfaction">
              <div className="metric-header">
                <div className="metric-icon">⭐</div>
                <span className="metric-label">Satisfaction</span>
              </div>
              <div className="metric-value">
                {data.customerMetrics.satisfaction}/5
              </div>
              <div className="metric-description">
                Average satisfaction score
              </div>
            </div>

            <div className="customer-metric-card churn">
              <div className="metric-header">
                <div className="metric-icon">📉</div>
                <span className="metric-label">Churn Rate</span>
              </div>
              <div className="metric-value">
                {data.customerMetrics.churnRate}%
              </div>
              <div className="metric-description">
                Customer churn this period
              </div>
            </div>

            <div className="customer-metric-card lifetime">
              <div className="metric-header">
                <div className="metric-icon">💎</div>
                <span className="metric-label">Lifetime Value</span>
              </div>
              <div className="metric-value">
                ${data.customerMetrics.lifetimeValue}
              </div>
              <div className="metric-description">Average customer LTV</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
