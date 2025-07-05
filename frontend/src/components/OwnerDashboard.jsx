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
} from "recharts";
import { useAuth } from "../context/AuthContext";
import { companyApi, dashboardApi, productsApi } from "../services/api";

const COLORS = ["#00d4ff", "#ff0080", "#8000ff", "#00ff88"];

export default function OwnerDashboard() {
  const { user } = useAuth();
  const [company, setCompany] = useState({});
  const [products, setProducts] = useState([]);
  const [stats, setStats] = useState({});
  const [loading, setLoading] = useState(true);
  const [timeFilter, setTimeFilter] = useState("30d");

  useEffect(() => {
    async function fetchData() {
      if (!user?.companyId) return;
      setLoading(true);
      try {
        const companyData = await companyApi.getCompany(user.companyId);
        const productsData = await productsApi.getCompanyProducts(user.companyId);
        const statsData = await dashboardApi.getCompanyStats(user.companyId, timeFilter);
        setCompany(companyData);
        setProducts(productsData);
        setStats(statsData);
      } catch (error) {
        console.error("Unexpected error in dashboard:", error);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, [user?.companyId, timeFilter]);

  const productsByCategory = products.reduce((acc, product) => {
    const category = product.category || "Other";
    acc[category] = (acc[category] || 0) + 1;
    return acc;
  }, {});

  const categoryChart = Object.entries(productsByCategory).map(
    ([name, value]) => ({ name, value }),
  );

  const recentProducts = products
    .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
    .slice(0, 5);

  if (loading) {
    return (
      <div className="dashboard-loading">
        <div className="cosmic-spinner-large"></div>
        <h2 className="text-xl font-semibold">Loading dashboard...</h2>
        <p className="text-base">Fetching your business insights</p>
      </div>
    );
  }

  return (
    <div className="owner-dashboard">
      <div className="dashboard-header">
        <div className="header-content">
          <div className="company-header">
            {company.logo && (
              <div className="company-logo">
                <img src={company.logo} alt={company.name} />
              </div>
            )}
            <div>
              <h1 className="text-3xl lg:text-4xl font-bold leading-tight">
                {company.name || "Your Company"}
              </h1>
              <p className="company-tagline text-base lg:text-lg leading-relaxed">
                {company.tagline || "Company Dashboard"}
              </p>
              <div className="company-meta text-sm lg:text-base">
                {company.industry && <span>{company.industry}</span>}
                {company.industry && company.country && <span> • </span>}
                {company.country && <span>{company.country}</span>}
                {company.founded_year && (
                  <span> • Est. {company.founded_year}</span>
                )}
              </div>
            </div>
          </div>
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
          </select>
        </div>
      </div>

      <div className="kpi-grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
        <div className="kpi-card primary">
          <div className="kpi-icon">📦</div>
          <div className="kpi-content">
            <h3 className="text-xs font-semibold">Products</h3>
            <div className="kpi-value text-2xl md:text-3xl font-bold">
              {products.length}
            </div>
            <div className="kpi-change positive text-sm">
              +{stats.newProducts || 0} this month
            </div>
          </div>
        </div>

        <div className="kpi-card success">
          <div className="kpi-icon">💰</div>
          <div className="kpi-content">
            <h3 className="text-xs font-semibold">Revenue</h3>
            <div className="kpi-value text-2xl md:text-3xl font-bold">
              $
              {typeof stats.revenue === "number"
                ? stats.revenue.toLocaleString()
                : stats.revenue || "0"}
            </div>
            <div className="kpi-change positive text-sm">
              +{stats.revenueGrowth || "0"}% this month
            </div>
          </div>
        </div>

        <div className="kpi-card warning">
          <div className="kpi-icon">🛍️</div>
          <div className="kpi-content">
            <h3 className="text-xs font-semibold">Orders</h3>
            <div className="kpi-value text-2xl md:text-3xl font-bold">
              {stats.orders || 0}
            </div>
            <div className="kpi-change positive text-sm">
              +{stats.newOrders || 0} this month
            </div>
          </div>
        </div>

        <div className="kpi-card info">
          <div className="kpi-icon">👥</div>
          <div className="kpi-content">
            <h3 className="text-xs font-semibold">Team Members</h3>
            <div className="kpi-value text-2xl md:text-3xl font-bold">
              {stats.teamMembers || 1}
            </div>
            <div className="kpi-change neutral text-sm">Active users</div>
          </div>
        </div>
      </div>

      <div className="dashboard-grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3">
        <div className="chart-card">
          <div className="chart-header">
            <h3 className="text-lg md:text-xl font-bold">
              📊 Product Categories
            </h3>
            <p className="text-sm md:text-base">
              Distribution of your products
            </p>
          </div>
          <div className="chart-container">
            {categoryChart.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={categoryChart}
                    dataKey="value"
                    nameKey="name"
                    outerRadius={100}
                    label={({ name, percent }) =>
                      `${name} ${(percent * 100).toFixed(0)}%`
                    }
                    labelLine={false}
                  >
                    {categoryChart.map((_, index) => (
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
            ) : (
              <div className="empty-chart">
                <div className="text-center p-lg">
                  <p className="text-base mb-lg">No products yet</p>
                  <button className="cosmic-btn primary">
                    Add your first product
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="chart-card">
          <div className="chart-header">
            <h3 className="text-lg md:text-xl font-bold">📈 Revenue Trend</h3>
            <p className="text-sm md:text-base">Monthly revenue over time</p>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={stats.revenueChart || []}>
                <XAxis
                  dataKey="month"
                  tick={{ fill: "#b3b3cc", fontSize: 12 }}
                  axisLine={{ stroke: "rgba(255, 255, 255, 0.1)" }}
                />
                <YAxis
                  tick={{ fill: "#b3b3cc", fontSize: 12 }}
                  axisLine={{ stroke: "rgba(255, 255, 255, 0.1)" }}
                />
                <Tooltip
                  formatter={(value) => [
                    `$${value.toLocaleString()}`,
                    "Revenue",
                  ]}
                  contentStyle={{
                    backgroundColor: "rgba(255, 255, 255, 0.05)",
                    border: "1px solid rgba(255, 255, 255, 0.1)",
                    borderRadius: "12px",
                    backdropFilter: "blur(20px)",
                    color: "#ffffff",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="revenue"
                  stroke="url(#revenueGradient)"
                  strokeWidth={3}
                  dot={{ fill: "#00d4ff", strokeWidth: 2, r: 4 }}
                />
                <defs>
                  <linearGradient
                    id="revenueGradient"
                    x1="0"
                    y1="0"
                    x2="1"
                    y2="0"
                  >
                    <stop offset="0%" stopColor="#00d4ff" />
                    <stop offset="100%" stopColor="#8000ff" />
                  </linearGradient>
                </defs>
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="chart-card">
          <div className="chart-header">
            <h3 className="text-lg md:text-xl font-bold">🏢 Company Profile</h3>
            <p className="text-sm md:text-base">Your company information</p>
          </div>
          <div className="company-profile">
            <div className="profile-item">
              <span className="profile-label text-sm">Industry:</span>
              <span className="profile-value text-sm">
                {company.industry || "Not specified"}
              </span>
            </div>
            <div className="profile-item">
              <span className="profile-label text-sm">Size:</span>
              <span className="profile-value text-sm">
                {company.size || "Not specified"}
              </span>
            </div>
            <div className="profile-item">
              <span className="profile-label text-sm">Location:</span>
              <span className="profile-value text-sm">
                {company.city
                  ? `${company.city}, ${company.country}`
                  : company.country || "Not specified"}
              </span>
            </div>
            <div className="profile-item">
              <span className="profile-label text-sm">Website:</span>
              <span className="profile-value text-sm">
                {company.website ? (
                  <a
                    href={company.website}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:text-primary-light transition-colors"
                  >
                    Visit Website
                  </a>
                ) : (
                  "Not specified"
                )}
              </span>
            </div>
            <div className="profile-item">
              <span className="profile-label text-sm">Contact:</span>
              <span className="profile-value text-sm">
                {company.contact_email || "Not specified"}
              </span>
            </div>
          </div>
        </div>

        <div className="chart-card md:col-span-2 xl:col-span-1">
          <div className="chart-header">
            <h3 className="text-lg md:text-xl font-bold">📦 Recent Products</h3>
            <p className="text-sm md:text-base">
              Latest additions to your catalog
            </p>
          </div>
          <div className="recent-products">
            {recentProducts.length > 0 ? (
              recentProducts.map((product) => (
                <div key={product.id} className="product-item">
                  <div className="product-info">
                    <div className="product-name text-sm font-semibold">
                      {product.name}
                    </div>
                    <div className="product-meta text-xs">
                      ${product.price} • Stock: {product.stock_qty || 0}
                    </div>
                  </div>
                  <div className="product-date text-xs">
                    {new Date(product.created_at).toLocaleDateString()}
                  </div>
                </div>
              ))
            ) : (
              <div className="empty-state p-lg">
                <div className="text-center">
                  <p className="text-sm mb-md">No products yet</p>
                  <button className="cosmic-btn primary text-sm">
                    Add Product
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
