import React, { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { companyApi } from "../services/api";
import { useTenant } from "./TenantContext";

const COLORS = [
  "#5f6fff",
  "#27ae60",
  "#e67e22",
  "#e74c3c",
  "#aab6ff",
  "#3b3bff",
];

export default function Dashboard() {
  const { tenant } = useTenant();
  const [companies, setCompanies] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchCompanies() {
      setLoading(true);
      try {
        const data = await companyApi.getCompanies();
        setCompanies(data);
      } catch {
        setCompanies([]);
      } finally {
        setLoading(false);
      }
    }
    fetchCompanies();
  }, []);

  // Infographics: Industry distribution, company sizes, countries
  const industryStats = Object.entries(
    companies.reduce((acc, c) => {
      if (!c.industry) return acc;
      acc[c.industry] = (acc[c.industry] || 0) + 1;
      return acc;
    }, {}),
  ).map(([name, value]) => ({ name, value }));

  const sizeStats = Object.entries(
    companies.reduce((acc, c) => {
      if (!c.size) return acc;
      acc[c.size] = (acc[c.size] || 0) + 1;
      return acc;
    }, {}),
  ).map(([name, value]) => ({ name, value }));

  const countryStats = Object.entries(
    companies.reduce((acc, c) => {
      if (!c.country) return acc;
      acc[c.country] = (acc[c.country] || 0) + 1;
      return acc;
    }, {}),
  ).map(([name, value]) => ({ name, value }));

  // Current tenant company details
  const current = companies.find((c) => c.id === tenant.id) || {};

  return (
    <>
      <h2>Dashboard</h2>
      <div style={{ marginBottom: 16, color: "#888", fontSize: 15 }}>
        <b>Tenant:</b> {tenant.name}
      </div>
      {loading ? (
        <div className="loading">Loading statistics…</div>
      ) : (
        <>
          <section className="kpi-cards">
            <div className="card">
              <h3>Industry</h3>
              <p>{current.industry || "-"}</p>
            </div>
            <div className="card">
              <h3>Size</h3>
              <p>{current.size || "-"}</p>
            </div>
            <div className="card">
              <h3>Country</h3>
              <p>{current.country || "-"}</p>
            </div>
            <div className="card">
              <h3>Founded</h3>
              <p>{current.founded_year || "-"}</p>
            </div>
            <div className="card">
              <h3>Website</h3>
              <p>
                {current.website ? (
                  <a
                    href={current.website}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Visit
                  </a>
                ) : (
                  "-"
                )}
              </p>
            </div>
          </section>

          <div
            style={{
              display: "flex",
              gap: 32,
              flexWrap: "wrap",
              marginBottom: 32,
            }}
          >
            <div className="chart-card">
              <h3 style={{ textAlign: "center", color: "var(--primary)" }}>
                Industry Distribution
              </h3>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie
                    data={industryStats}
                    dataKey="value"
                    nameKey="name"
                    outerRadius={80}
                    label
                  >
                    {industryStats.map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-card">
              <h3 style={{ textAlign: "center", color: "var(--success)" }}>
                Company Sizes
              </h3>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={sizeStats}>
                  <XAxis dataKey="name" />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="value" fill="var(--success)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-card">
              <h3 style={{ textAlign: "center", color: "var(--warning)" }}>
                Countries
              </h3>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={countryStats}>
                  <XAxis dataKey="name" />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="value" fill="var(--warning)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}
    </>
  );
}
