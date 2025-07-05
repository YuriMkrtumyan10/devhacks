import React, { useState, useEffect } from "react";
import { companyApi } from "../services/api";
import { useAuth } from "../context/AuthContext";
import { useTenant } from "./TenantContext";

export default function Company() {
  const { user } = useAuth();
  const { refreshTenants } = useTenant();
  const [companies, setCompanies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCompany, setSelectedCompany] = useState(null);
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [newCompany, setNewCompany] = useState({
    name: "",
    industry: "",
    country: "",
    city: "",
    website: "",
    contact_email: "",
    phone: "",
    tagline: "",
    founded_year: new Date().getFullYear(),
    size: "1-10",
  });

  useEffect(() => {
    fetchCompanies();
  }, []);

  const fetchCompanies = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await companyApi.getCompanies();
      setCompanies(data);
    } catch (err) {
      setError(err.message);
      setCompanies([]);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateCompany = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      await companyApi.createCompany(newCompany);
      setIsCreateModalOpen(false);
      setNewCompany({
        name: "",
        industry: "",
        country: "",
        city: "",
        website: "",
        contact_email: "",
        phone: "",
        tagline: "",
        founded_year: new Date().getFullYear(),
        size: "1-10",
      });
      await fetchCompanies();
      await refreshTenants();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteCompany = async (companyId) => {
    if (!window.confirm("Are you sure you want to delete this company?"))
      return;

    try {
      await companyApi.deleteCompany(companyId);
      await fetchCompanies();
      await refreshTenants();
    } catch (err) {
      setError(err.message);
    }
  };

  const filteredCompanies = companies.filter(
    (company) =>
      company.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (company.industry || "")
        .toLowerCase()
        .includes(searchTerm.toLowerCase()) ||
      (company.country || "").toLowerCase().includes(searchTerm.toLowerCase()),
  );

  if (loading && companies.length === 0) {
    return (
      <div className="cosmic-loading">
        <div className="cosmic-spinner-large"></div>
        <h2>Loading companies...</h2>
        <p>Fetching data from the cosmic database</p>
      </div>
    );
  }

  return (
    <div className="cosmic-companies">
      <div className="companies-header">
        <div className="header-content">
          <h1>🏢 Company Management</h1>
          <p>Manage and oversee all companies in the cosmic network</p>
        </div>

        <div className="header-actions">
          <div className="cosmic-search-bar">
            <svg
              className="search-icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
            >
              <circle cx="11" cy="11" r="8"></circle>
              <path d="21 21l-4.35-4.35"></path>
            </svg>
            <input
              type="text"
              placeholder="Search companies, industries, countries..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
          </div>

          <button
            className="cosmic-btn primary"
            onClick={() => setIsCreateModalOpen(true)}
          >
            <span className="btn-icon">✨</span>
            Add Company
          </button>
        </div>
      </div>

      {error && (
        <div className="cosmic-alert error">
          <span className="alert-icon">⚠️</span>
          <div className="alert-content">
            <h4>Error</h4>
            <p>{error}</p>
          </div>
        </div>
      )}

      <div className="companies-stats">
        <div className="stat-card">
          <div className="stat-icon">🏢</div>
          <div className="stat-info">
            <h3>{companies.length}</h3>
            <p>Total Companies</p>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">🌍</div>
          <div className="stat-info">
            <h3>
              {new Set(companies.map((c) => c.country).filter(Boolean)).size}
            </h3>
            <p>Countries</p>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">🏭</div>
          <div className="stat-info">
            <h3>
              {new Set(companies.map((c) => c.industry).filter(Boolean)).size}
            </h3>
            <p>Industries</p>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">📈</div>
          <div className="stat-info">
            <h3>
              {
                companies.filter(
                  (c) =>
                    new Date(c.created_at) >
                    new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
                ).length
              }
            </h3>
            <p>New This Month</p>
          </div>
        </div>
      </div>

      <div className="companies-grid">
        {filteredCompanies.map((company) => (
          <div key={company.id} className="company-card">
            <div className="company-header">
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
                <h3 className="company-name">{company.name}</h3>
                <p className="company-tagline">
                  {company.tagline || "Building the future"}
                </p>
              </div>
              <div className="company-actions">
                <button
                  className="action-btn edit"
                  onClick={() => setSelectedCompany(company)}
                  title="Edit Company"
                >
                  ✏️
                </button>
                <button
                  className="action-btn delete"
                  onClick={() => handleDeleteCompany(company.id)}
                  title="Delete Company"
                >
                  🗑️
                </button>
              </div>
            </div>

            <div className="company-details">
              <div className="detail-row">
                <span className="detail-label">Industry:</span>
                <span className="detail-value">
                  {company.industry || "Not specified"}
                </span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Location:</span>
                <span className="detail-value">
                  {company.city
                    ? `${company.city}, ${company.country}`
                    : company.country || "Not specified"}
                </span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Founded:</span>
                <span className="detail-value">
                  {company.founded_year || "Unknown"}
                </span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Size:</span>
                <span className="detail-value">
                  {company.size || "Not specified"}
                </span>
              </div>
            </div>

            {company.website && (
              <div className="company-footer">
                <a
                  href={company.website}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="website-link"
                >
                  🌐 Visit Website
                </a>
              </div>
            )}
          </div>
        ))}
      </div>

      {filteredCompanies.length === 0 && !loading && (
        <div className="cosmic-empty-state">
          <div className="empty-icon">🔍</div>
          <h3>No companies found</h3>
          <p>
            {searchTerm
              ? `No companies match "${searchTerm}". Try adjusting your search.`
              : "No companies have been added yet. Create your first company to get started."}
          </p>
          {!searchTerm && (
            <button
              className="cosmic-btn primary"
              onClick={() => setIsCreateModalOpen(true)}
            >
              <span className="btn-icon">✨</span>
              Create First Company
            </button>
          )}
        </div>
      )}

      {/* Create Company Modal */}
      {isCreateModalOpen && (
        <div className="cosmic-modal">
          <div
            className="modal-backdrop"
            onClick={() => setIsCreateModalOpen(false)}
          ></div>
          <div className="modal-content">
            <div className="modal-header">
              <h2>✨ Create New Company</h2>
              <button
                className="close-btn"
                onClick={() => setIsCreateModalOpen(false)}
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleCreateCompany} className="company-form">
              <div className="form-grid">
                <div className="form-group">
                  <label htmlFor="name">Company Name *</label>
                  <input
                    id="name"
                    type="text"
                    value={newCompany.name}
                    onChange={(e) =>
                      setNewCompany({ ...newCompany, name: e.target.value })
                    }
                    required
                    placeholder="Enter company name"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="industry">Industry</label>
                  <select
                    id="industry"
                    value={newCompany.industry}
                    onChange={(e) =>
                      setNewCompany({ ...newCompany, industry: e.target.value })
                    }
                  >
                    <option value="">Select industry</option>
                    <option value="Technology">Technology</option>
                    <option value="Healthcare">Healthcare</option>
                    <option value="Finance">Finance</option>
                    <option value="E-commerce">E-commerce</option>
                    <option value="Manufacturing">Manufacturing</option>
                    <option value="Education">Education</option>
                    <option value="Real Estate">Real Estate</option>
                    <option value="Other">Other</option>
                  </select>
                </div>

                <div className="form-group">
                  <label htmlFor="country">Country</label>
                  <input
                    id="country"
                    type="text"
                    value={newCompany.country}
                    onChange={(e) =>
                      setNewCompany({ ...newCompany, country: e.target.value })
                    }
                    placeholder="e.g., United States"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="city">City</label>
                  <input
                    id="city"
                    type="text"
                    value={newCompany.city}
                    onChange={(e) =>
                      setNewCompany({ ...newCompany, city: e.target.value })
                    }
                    placeholder="e.g., San Francisco"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="website">Website</label>
                  <input
                    id="website"
                    type="url"
                    value={newCompany.website}
                    onChange={(e) =>
                      setNewCompany({ ...newCompany, website: e.target.value })
                    }
                    placeholder="https://example.com"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="contact_email">Contact Email</label>
                  <input
                    id="contact_email"
                    type="email"
                    value={newCompany.contact_email}
                    onChange={(e) =>
                      setNewCompany({
                        ...newCompany,
                        contact_email: e.target.value,
                      })
                    }
                    placeholder="contact@company.com"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="phone">Phone</label>
                  <input
                    id="phone"
                    type="tel"
                    value={newCompany.phone}
                    onChange={(e) =>
                      setNewCompany({ ...newCompany, phone: e.target.value })
                    }
                    placeholder="+1 (555) 123-4567"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="size">Company Size</label>
                  <select
                    id="size"
                    value={newCompany.size}
                    onChange={(e) =>
                      setNewCompany({ ...newCompany, size: e.target.value })
                    }
                  >
                    <option value="1-10">1-10 employees</option>
                    <option value="11-50">11-50 employees</option>
                    <option value="51-200">51-200 employees</option>
                    <option value="201-500">201-500 employees</option>
                    <option value="501-1000">501-1000 employees</option>
                    <option value="1000+">1000+ employees</option>
                  </select>
                </div>
              </div>

              <div className="form-group full-width">
                <label htmlFor="tagline">Tagline</label>
                <input
                  id="tagline"
                  type="text"
                  value={newCompany.tagline}
                  onChange={(e) =>
                    setNewCompany({ ...newCompany, tagline: e.target.value })
                  }
                  placeholder="A brief description of what the company does"
                />
              </div>

              <div className="modal-actions">
                <button
                  type="button"
                  className="cosmic-btn secondary"
                  onClick={() => setIsCreateModalOpen(false)}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="cosmic-btn primary"
                  disabled={loading || !newCompany.name.trim()}
                >
                  {loading ? (
                    <>
                      <div className="cosmic-spinner"></div>
                      Creating...
                    </>
                  ) : (
                    <>
                      <span className="btn-icon">✨</span>
                      Create Company
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
