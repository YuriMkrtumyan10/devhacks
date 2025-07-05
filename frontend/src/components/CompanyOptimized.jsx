import React, { useState, useEffect } from "react";
import { companyApi } from "../services/api";
import { useAuth } from "../context/AuthContext";
import { useTenant } from "./TenantContext";

export default function CompanyOptimized() {
  const { user } = useAuth();
  const { refreshTenants } = useTenant();
  const [companies, setCompanies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCompany, setSelectedCompany] = useState(null);
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [editingCompany, setEditingCompany] = useState(null);
  const [operationLoading, setOperationLoading] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [companyToDelete, setCompanyToDelete] = useState(null);
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
    setOperationLoading(true);
    setError("");
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
      setError(`Failed to create company: ${err.message}`);
    } finally {
      setOperationLoading(false);
    }
  };

  const handleEditCompany = (company) => {
    setEditingCompany({
      id: company.id,
      name: company.name || "",
      industry: company.industry || "",
      country: company.country || "",
      city: company.city || "",
      website: company.website || "",
      contact_email: company.contact_email || "",
      phone: company.phone || "",
      tagline: company.tagline || "",
      founded_year: company.founded_year || new Date().getFullYear(),
      size: company.size || "1-10",
    });
    setIsEditModalOpen(true);
    setSelectedCompany(null);
  };

  const handleUpdateCompany = async (e) => {
    e.preventDefault();
    setOperationLoading(true);
    setError("");
    try {
      await companyApi.updateCompany(editingCompany.id, editingCompany);
      setIsEditModalOpen(false);
      setEditingCompany(null);
      await fetchCompanies();
      await refreshTenants();
    } catch (err) {
      setError(`Failed to update company: ${err.message}`);
    } finally {
      setOperationLoading(false);
    }
  };

  const handleDeleteClick = (company) => {
    setCompanyToDelete(company);
    setIsDeleteModalOpen(true);
  };

  const handleConfirmDelete = async () => {
    if (!companyToDelete) return;

    setOperationLoading(true);
    setError("");
    try {
      await companyApi.deleteCompany(companyToDelete.id);
      await fetchCompanies();
      await refreshTenants();
      setIsDeleteModalOpen(false);
      setCompanyToDelete(null);
    } catch (err) {
      setError(`Failed to delete company: ${err.message}`);
    } finally {
      setOperationLoading(false);
    }
  };

  const filteredCompanies = companies.filter(
    (company) =>
      (company.name || "").toLowerCase().includes(searchTerm.toLowerCase()) ||
      (company.industry || "")
        .toLowerCase()
        .includes(searchTerm.toLowerCase()) ||
      (company.country || "")
        .toLowerCase()
        .includes(searchTerm.toLowerCase()) ||
      (company.city || "").toLowerCase().includes(searchTerm.toLowerCase()) ||
      (company.tagline || "").toLowerCase().includes(searchTerm.toLowerCase()),
  );

  if (loading && companies.length === 0) {
    return (
      <div className="cosmic-loading">
        <div className="cosmic-spinner-large"></div>
        <h2 className="text-xl lg:text-2xl font-semibold">
          Loading companies...
        </h2>
        <p className="text-base lg:text-lg">Fetching data from the database</p>
      </div>
    );
  }

  return (
    <div className="cosmic-companies">
      <div className="companies-header">
        <div className="header-content">
          <h1 className="text-3xl lg:text-4xl font-bold leading-tight">
            🏢 Company Management
          </h1>
          <p className="text-base lg:text-lg leading-relaxed">
            Manage and oversee all companies in the network
          </p>
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
            <span className="hidden sm:inline">Add Company</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="cosmic-alert error">
          <span className="alert-icon">⚠️</span>
          <div className="alert-content">
            <h4 className="font-semibold">Error</h4>
            <p className="text-sm">{error}</p>
          </div>
        </div>
      )}

      <div className="companies-stats grid-cols-2 lg:grid-cols-4">
        <div className="stat-card">
          <div className="stat-icon">🏢</div>
          <div className="stat-info">
            <h3 className="text-xl lg:text-2xl font-bold">
              {companies.length}
            </h3>
            <p className="text-sm lg:text-base">Total Companies</p>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">🌍</div>
          <div className="stat-info">
            <h3 className="text-xl lg:text-2xl font-bold">
              {new Set(companies.map((c) => c.country).filter(Boolean)).size}
            </h3>
            <p className="text-sm lg:text-base">Countries</p>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">🏭</div>
          <div className="stat-info">
            <h3 className="text-xl lg:text-2xl font-bold">
              {new Set(companies.map((c) => c.industry).filter(Boolean)).size}
            </h3>
            <p className="text-sm lg:text-base">Industries</p>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">📈</div>
          <div className="stat-info">
            <h3 className="text-xl lg:text-2xl font-bold">
              {
                companies.filter(
                  (c) =>
                    new Date(c.created_at) >
                    new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
                ).length
              }
            </h3>
            <p className="text-sm lg:text-base">New This Month</p>
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
                    {(company.name || "C").charAt(0)}
                  </div>
                )}
              </div>
              <div className="company-info">
                <h3 className="company-name text-lg font-bold">
                  {company.name}
                </h3>
                <p className="company-tagline text-sm">
                  {company.tagline || "Building the future"}
                </p>
              </div>
              <div className="company-actions">
                <button
                  className="action-btn edit"
                  onClick={() => handleEditCompany(company)}
                  title="Edit Company"
                  aria-label={`Edit ${company.name}`}
                  disabled={operationLoading}
                >
                  ✏️
                </button>
                <button
                  className="action-btn delete"
                  onClick={() => handleDeleteClick(company)}
                  title="Delete Company"
                  aria-label={`Delete ${company.name}`}
                  disabled={operationLoading}
                >
                  🗑️
                </button>
              </div>
            </div>

            <div className="company-details">
              <div className="detail-row">
                <span className="detail-label text-xs">Industry:</span>
                <span className="detail-value text-xs">
                  {company.industry || "Not specified"}
                </span>
              </div>
              <div className="detail-row">
                <span className="detail-label text-xs">Location:</span>
                <span className="detail-value text-xs">
                  {company.city
                    ? `${company.city}, ${company.country}`
                    : company.country || "Not specified"}
                </span>
              </div>
              <div className="detail-row">
                <span className="detail-label text-xs">Founded:</span>
                <span className="detail-value text-xs">
                  {company.founded_year || "Unknown"}
                </span>
              </div>
              <div className="detail-row">
                <span className="detail-label text-xs">Size:</span>
                <span className="detail-value text-xs">
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
                  className="website-link text-sm"
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
          <h3 className="text-xl lg:text-2xl font-semibold">
            No companies found
          </h3>
          <p className="text-base lg:text-lg">
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
              <h2 className="text-xl lg:text-2xl font-bold">
                ✨ Create New Company
              </h2>
              <button
                className="close-btn"
                onClick={() => setIsCreateModalOpen(false)}
                aria-label="Close modal"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleCreateCompany} className="company-form">
              <div className="form-grid">
                <div className="form-group">
                  <label htmlFor="name" className="text-sm font-semibold">
                    Company Name *
                  </label>
                  <input
                    id="name"
                    type="text"
                    value={newCompany.name}
                    onChange={(e) =>
                      setNewCompany({ ...newCompany, name: e.target.value })
                    }
                    required
                    placeholder="Enter company name"
                    className="text-base"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="industry" className="text-sm font-semibold">
                    Industry
                  </label>
                  <select
                    id="industry"
                    value={newCompany.industry}
                    onChange={(e) =>
                      setNewCompany({ ...newCompany, industry: e.target.value })
                    }
                    className="text-base"
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
                  <label htmlFor="country" className="text-sm font-semibold">
                    Country
                  </label>
                  <input
                    id="country"
                    type="text"
                    value={newCompany.country}
                    onChange={(e) =>
                      setNewCompany({ ...newCompany, country: e.target.value })
                    }
                    placeholder="e.g., United States"
                    className="text-base"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="city" className="text-sm font-semibold">
                    City
                  </label>
                  <input
                    id="city"
                    type="text"
                    value={newCompany.city}
                    onChange={(e) =>
                      setNewCompany({ ...newCompany, city: e.target.value })
                    }
                    placeholder="e.g., San Francisco"
                    className="text-base"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="website" className="text-sm font-semibold">
                    Website
                  </label>
                  <input
                    id="website"
                    type="url"
                    value={newCompany.website}
                    onChange={(e) =>
                      setNewCompany({ ...newCompany, website: e.target.value })
                    }
                    placeholder="https://example.com"
                    className="text-base"
                  />
                </div>

                <div className="form-group">
                  <label
                    htmlFor="contact_email"
                    className="text-sm font-semibold"
                  >
                    Contact Email
                  </label>
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
                    className="text-base"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="phone" className="text-sm font-semibold">
                    Phone
                  </label>
                  <input
                    id="phone"
                    type="tel"
                    value={newCompany.phone}
                    onChange={(e) =>
                      setNewCompany({ ...newCompany, phone: e.target.value })
                    }
                    placeholder="+1 (555) 123-4567"
                    className="text-base"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="size" className="text-sm font-semibold">
                    Company Size
                  </label>
                  <select
                    id="size"
                    value={newCompany.size}
                    onChange={(e) =>
                      setNewCompany({ ...newCompany, size: e.target.value })
                    }
                    className="text-base"
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
                <label htmlFor="tagline" className="text-sm font-semibold">
                  Tagline
                </label>
                <input
                  id="tagline"
                  type="text"
                  value={newCompany.tagline}
                  onChange={(e) =>
                    setNewCompany({ ...newCompany, tagline: e.target.value })
                  }
                  placeholder="A brief description of what the company does"
                  className="text-base"
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
                  disabled={operationLoading || !newCompany.name.trim()}
                >
                  {operationLoading ? (
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

      {/* Edit Company Modal */}
      {isEditModalOpen && editingCompany && (
        <div className="cosmic-modal">
          <div
            className="modal-backdrop"
            onClick={() => setIsEditModalOpen(false)}
          ></div>
          <div className="modal-content">
            <div className="modal-header">
              <h2 className="text-xl lg:text-2xl font-bold">✏️ Edit Company</h2>
              <button
                className="close-btn"
                onClick={() => setIsEditModalOpen(false)}
                aria-label="Close modal"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleUpdateCompany} className="company-form">
              <div className="form-grid">
                <div className="form-group">
                  <label htmlFor="edit-name" className="text-sm font-semibold">
                    Company Name *
                  </label>
                  <input
                    id="edit-name"
                    type="text"
                    value={editingCompany.name}
                    onChange={(e) =>
                      setEditingCompany({
                        ...editingCompany,
                        name: e.target.value,
                      })
                    }
                    required
                    placeholder="Enter company name"
                    className="text-base"
                  />
                </div>

                <div className="form-group">
                  <label
                    htmlFor="edit-industry"
                    className="text-sm font-semibold"
                  >
                    Industry
                  </label>
                  <select
                    id="edit-industry"
                    value={editingCompany.industry}
                    onChange={(e) =>
                      setEditingCompany({
                        ...editingCompany,
                        industry: e.target.value,
                      })
                    }
                    className="text-base"
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
                  <label
                    htmlFor="edit-country"
                    className="text-sm font-semibold"
                  >
                    Country
                  </label>
                  <input
                    id="edit-country"
                    type="text"
                    value={editingCompany.country}
                    onChange={(e) =>
                      setEditingCompany({
                        ...editingCompany,
                        country: e.target.value,
                      })
                    }
                    placeholder="e.g., United States"
                    className="text-base"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="edit-city" className="text-sm font-semibold">
                    City
                  </label>
                  <input
                    id="edit-city"
                    type="text"
                    value={editingCompany.city}
                    onChange={(e) =>
                      setEditingCompany({
                        ...editingCompany,
                        city: e.target.value,
                      })
                    }
                    placeholder="e.g., San Francisco"
                    className="text-base"
                  />
                </div>

                <div className="form-group">
                  <label
                    htmlFor="edit-website"
                    className="text-sm font-semibold"
                  >
                    Website
                  </label>
                  <input
                    id="edit-website"
                    type="url"
                    value={editingCompany.website}
                    onChange={(e) =>
                      setEditingCompany({
                        ...editingCompany,
                        website: e.target.value,
                      })
                    }
                    placeholder="https://example.com"
                    className="text-base"
                  />
                </div>

                <div className="form-group">
                  <label
                    htmlFor="edit-contact_email"
                    className="text-sm font-semibold"
                  >
                    Contact Email
                  </label>
                  <input
                    id="edit-contact_email"
                    type="email"
                    value={editingCompany.contact_email}
                    onChange={(e) =>
                      setEditingCompany({
                        ...editingCompany,
                        contact_email: e.target.value,
                      })
                    }
                    placeholder="contact@company.com"
                    className="text-base"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="edit-phone" className="text-sm font-semibold">
                    Phone
                  </label>
                  <input
                    id="edit-phone"
                    type="tel"
                    value={editingCompany.phone}
                    onChange={(e) =>
                      setEditingCompany({
                        ...editingCompany,
                        phone: e.target.value,
                      })
                    }
                    placeholder="+1 (555) 123-4567"
                    className="text-base"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="edit-size" className="text-sm font-semibold">
                    Company Size
                  </label>
                  <select
                    id="edit-size"
                    value={editingCompany.size}
                    onChange={(e) =>
                      setEditingCompany({
                        ...editingCompany,
                        size: e.target.value,
                      })
                    }
                    className="text-base"
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
                <label htmlFor="edit-tagline" className="text-sm font-semibold">
                  Tagline
                </label>
                <input
                  id="edit-tagline"
                  type="text"
                  value={editingCompany.tagline}
                  onChange={(e) =>
                    setEditingCompany({
                      ...editingCompany,
                      tagline: e.target.value,
                    })
                  }
                  placeholder="A brief description of what the company does"
                  className="text-base"
                />
              </div>

              <div className="modal-actions">
                <button
                  type="button"
                  className="cosmic-btn secondary"
                  onClick={() => setIsEditModalOpen(false)}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="cosmic-btn primary"
                  disabled={operationLoading || !editingCompany.name.trim()}
                >
                  {operationLoading ? (
                    <>
                      <div className="cosmic-spinner"></div>
                      Updating...
                    </>
                  ) : (
                    <>
                      <span className="btn-icon">✏️</span>
                      Update Company
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {isDeleteModalOpen && companyToDelete && (
        <div className="cosmic-modal">
          <div
            className="modal-backdrop"
            onClick={() => setIsDeleteModalOpen(false)}
          ></div>
          <div className="modal-content">
            <div className="modal-header">
              <h2 className="text-xl lg:text-2xl font-bold">
                🗑️ Delete Company
              </h2>
              <button
                className="close-btn"
                onClick={() => setIsDeleteModalOpen(false)}
                aria-label="Close modal"
              >
                ✕
              </button>
            </div>

            <div className="modal-body">
              <p className="text-base lg:text-lg mb-4">
                Are you sure you want to delete{" "}
                <strong>{companyToDelete.name}</strong>?
              </p>
              <p className="text-sm text-gray-600">
                This action cannot be undone. All company data will be
                permanently removed.
              </p>
            </div>

            <div className="modal-actions">
              <button
                type="button"
                className="cosmic-btn secondary"
                onClick={() => setIsDeleteModalOpen(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="cosmic-btn danger"
                onClick={handleConfirmDelete}
                disabled={operationLoading}
              >
                {operationLoading ? (
                  <>
                    <div className="cosmic-spinner"></div>
                    Deleting...
                  </>
                ) : (
                  <>
                    <span className="btn-icon">🗑️</span>
                    Delete Company
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
