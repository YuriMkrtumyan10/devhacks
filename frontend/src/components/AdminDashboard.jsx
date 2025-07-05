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
import { companyApi, dashboardApi } from "../services/api";

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

export default function AdminDashboard() {
  const [companies, setCompanies] = useState([]);
  const [stats, setStats] = useState({});
  const [loading, setLoading] = useState(true);
  const [timeFilter, setTimeFilter] = useState("30d");

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      try {
        const [companiesData, statsData] = await Promise.all([
          companyApi.getCompanies(),
          dashboardApi.getAdminStats(timeFilter),
        ]);
        setCompanies(companiesData);
        setStats(statsData);
      } catch (error) {
        console.error("Failed to fetch dashboard data:", error);
        setCompanies([]);
        setStats({});
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, [timeFilter]);

  const industryData = companies.reduce((acc, company) => {
    const industry = company.industry || "Other";
    acc[industry] = (acc[industry] || 0) + 1;
    return acc;
  }, {});

  const sizeData = companies.reduce((acc, company) => {
    const size = company.size || "Unknown";
    acc[size] = (acc[size] || 0) + 1;
    return acc;
  }, {});

  const countryData = companies.reduce((acc, company) => {
    const country = company.country || "Unknown";
    acc[country] = (acc[country] || 0) + 1;
    return acc;
  }, {});

  const industryChart = Object.entries(industryData).map(([name, value]) => ({
    name,
    value,
  }));
  const sizeChart = Object.entries(sizeData).map(([name, value]) => ({
    name,
    value,
  }));
  const countryChart = Object.entries(countryData)
    .map(([name, value]) => ({ name, value }))
    .slice(0, 8);

  const recentCompanies = companies
    .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
    .slice(0, 5);

  if (loading) {
    return (
      <div className="dashboard-loading">
        <div className="loading-spinner"></div>
        <p>Loading admin dashboard...</p>
      </div>
    );
  }

  return (
    <div className="admin-dashboard">
      <div className="dashboard-header">
        <div className="header-content">
          <h1 className="text-3xl lg:text-4xl font-bold leading-tight">
            🚀 Super Admin Dashboard
          </h1>
          <p className="text-base lg:text-lg leading-relaxed">
            Manage and monitor all companies across the platform
          </p>
        </div>
        <div className="header-actions">
          <select
            value={timeFilter}
            onChange={(e) => setTimeFilter(e.target.value)}
            className="time-filter"
          >
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="90d">Last 90 days</option>
            <option value="1y">Last year</option>
          </select>
        </div>
      </div>

      <div className="kpi-grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
        <div className="kpi-card primary">
          <div className="kpi-icon">🏢</div>
          <div className="kpi-content">
            <h3 className="text-xs font-semibold">Total Companies</h3>
            <div className="kpi-value text-2xl md:text-3xl font-bold">
              {companies.length}
            </div>
            <div className="kpi-change positive text-sm">
              +{stats.newCompanies || 0} this month
            </div>
          </div>
        </div>

        <div className="kpi-card success">
          <div className="kpi-icon">👥</div>
          <div className="kpi-content">
            <h3 className="text-xs font-semibold">Total Users</h3>
            <div className="kpi-value text-2xl md:text-3xl font-bold">
              {stats.totalUsers || 0}
            </div>
            <div className="kpi-change positive text-sm">
              +{stats.newUsers || 0} this month
            </div>
          </div>
        </div>

        <div className="kpi-card warning">
          <div className="kpi-icon">📦</div>
          <div className="kpi-content">
            <h3 className="text-xs font-semibold">Total Products</h3>
            <div className="kpi-value text-2xl md:text-3xl font-bold">
              {stats.totalProducts || 0}
            </div>
            <div className="kpi-change positive text-sm">
              +{stats.newProducts || 0} this month
            </div>
          </div>
        </div>

        <div className="kpi-card info">
          <div className="kpi-icon">🌍</div>
          <div className="kpi-content">
            <h3 className="text-xs font-semibold">Countries</h3>
            <div className="kpi-value text-2xl md:text-3xl font-bold">
              {Object.keys(countryData).length}
            </div>
            <div className="kpi-change neutral text-sm">Global presence</div>
          </div>
        </div>
      </div>

      <div className="dashboard-grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3">
        <div className="chart-card">
          <div className="chart-header">
            <h3 className="text-lg md:text-xl font-bold">
              📊 Industry Distribution
            </h3>
            <p className="text-sm md:text-base">Companies by industry sector</p>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={industryChart}
                  dataKey="value"
                  nameKey="name"
                  outerRadius={100}
                  label={({ name, percent }) =>
                    `${name} ${(percent * 100).toFixed(0)}%`
                  }
                  labelLine={false}
                >
                  {industryChart.map((_, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={COLORS[index % COLORS.length]}
                    />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(255, 255, 255, 0.05)",
                    border: "1px solid rgba(255, 255, 255, 0.1)",
                    borderRadius: "12px",
                    backdropFilter: "blur(20px)",
                    color: "#ffffff",
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="chart-card">
          <div className="chart-header">
            <h3 className="text-lg md:text-xl font-bold">👥 Company Sizes</h3>
            <p className="text-sm md:text-base">
              Distribution by employee count
            </p>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={sizeChart}>
                <XAxis
                  dataKey="name"
                  tick={{ fill: "#b3b3cc", fontSize: 12 }}
                  axisLine={{ stroke: "rgba(255, 255, 255, 0.1)" }}
                />
                <YAxis
                  tick={{ fill: "#b3b3cc", fontSize: 12 }}
                  axisLine={{ stroke: "rgba(255, 255, 255, 0.1)" }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(255, 255, 255, 0.05)",
                    border: "1px solid rgba(255, 255, 255, 0.1)",
                    borderRadius: "12px",
                    backdropFilter: "blur(20px)",
                    color: "#ffffff",
                  }}
                />
                <Bar
                  dataKey="value"
                  fill="url(#blueGradient)"
                  radius={[4, 4, 0, 0]}
                />
                <defs>
                  <linearGradient id="blueGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#00d4ff" />
                    <stop offset="100%" stopColor="#8000ff" />
                  </linearGradient>
                </defs>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="chart-card">
          <div className="chart-header">
            <h3 className="text-lg md:text-xl font-bold">
              🌍 Geographic Distribution
            </h3>
            <p className="text-sm md:text-base">Companies by country</p>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={countryChart}>
                <XAxis
                  dataKey="name"
                  tick={{ fill: "#b3b3cc", fontSize: 12 }}
                  axisLine={{ stroke: "rgba(255, 255, 255, 0.1)" }}
                />
                <YAxis
                  tick={{ fill: "#b3b3cc", fontSize: 12 }}
                  axisLine={{ stroke: "rgba(255, 255, 255, 0.1)" }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(255, 255, 255, 0.05)",
                    border: "1px solid rgba(255, 255, 255, 0.1)",
                    borderRadius: "12px",
                    backdropFilter: "blur(20px)",
                    color: "#ffffff",
                  }}
                />
                <Bar
                  dataKey="value"
                  fill="url(#greenGradient)"
                  radius={[4, 4, 0, 0]}
                />
                <defs>
                  <linearGradient
                    id="greenGradient"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop offset="0%" stopColor="#00ff88" />
                    <stop offset="100%" stopColor="#ff0080" />
                  </linearGradient>
                </defs>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="chart-card md:col-span-2 xl:col-span-1">
          <div className="chart-header">
            <h3 className="text-lg md:text-xl font-bold">
              🏢 Recent Companies
            </h3>
            <p className="text-sm md:text-base">Latest company registrations</p>
          </div>
          <div className="recent-companies">
            {recentCompanies.length > 0 ? (
              recentCompanies.map((company) => (
                <div key={company.id} className="company-item">
                  <div className="company-avatar">
                    {company.logo ? (
                      <img src={company.logo} alt={company.name} />
                    ) : (
                      <div className="company-initial">
                        {company.name.charAt(0)}
                      </div>
                    )}
                  </div>
                  <div className="company-info">
                    <div className="company-name text-sm font-semibold">
                      {company.name}
                    </div>
                    <div className="company-meta text-xs">
                      {company.industry && <span>{company.industry}</span>}
                      {company.industry && company.country && <span> • </span>}
                      {company.country && <span>{company.country}</span>}
                    </div>
                  </div>
                  <div className="company-date text-xs">
                    {new Date(company.created_at).toLocaleDateString()}
                  </div>
                </div>
              ))
            ) : (
              <div className="empty-state p-lg">
                <div className="text-center">
                  <p className="text-sm">No recent companies</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
