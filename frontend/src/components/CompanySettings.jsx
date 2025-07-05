// src/pages/CompanySettings.jsx
import React, { useEffect, useState } from "react";
import { companyApi } from "../services/api";
import { useTenant } from "./TenantContext";

/**
 * Company Settings & Privacy page
 *
 * Props:
 *   - companyId (number)  Optional. Defaults to 1.
 */
export default function CompanySettings({ companyId = 1 }) {
  const { tenant } = useTenant();
  // ---------- STATE ----------
  const [form, setForm] = useState({
    companyName: "",
    contactEmail: "",
    phoneNumber: "",
    description: "",
    privacyPolicy: "",
  });
  const [status, setStatus] = useState("idle"); // idle | loading | saving | success | error
  const [error, setError] = useState(null);

  // ---------- FETCH ON MOUNT ----------
  useEffect(() => {
    const fetchCompany = async () => {
      try {
        setStatus("loading");
        const data = await companyApi.getCompany(companyId);
        // Map backend fields to frontend form fields
        setForm({
          companyName: data.name ?? data.companyName ?? "",
          contactEmail: data.contact_email ?? data.contactEmail ?? "",
          phoneNumber: data.phone ?? data.phoneNumber ?? "",
          description: data.description ?? data.details ?? "",
          privacyPolicy: data.privacy_policy ?? data.privacyPolicy ?? "",
        });
        setStatus("idle");
      } catch (err) {
        setError(err.message);
        setStatus("error");
      }
    };

    fetchCompany();
  }, [companyId]);

  // ---------- HANDLERS ----------
  const handleChange = (e) => {
    const { name, type, checked, value } = e.target;
    setForm((f) => ({ ...f, [name]: type === "checkbox" ? checked : value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      setStatus("saving");
      // Map frontend fields to backend fields for update
      await companyApi.updateCompany(companyId, {
        name: form.companyName,
        contact_email: form.contactEmail,
        phone: form.phoneNumber,
        description: form.description,
        privacy_policy: form.privacyPolicy,
      });
      setStatus("success");
      setTimeout(() => setStatus("idle"), 2500);
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  };

  // ---------- RENDER ----------
  if (status === "loading")
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <span>Loading company data...</span>
      </div>
    );

  if (status === "error")
    return (
      <div className="error-container">
        <span className="error-icon">⚠️</span>
        <span>Error: {error}</span>
      </div>
    );

  return (
    <section className="company-settings">
      <div className="settings-header">
        <h2 className="settings-title">Company Settings & Privacy</h2>
        <div className="tenant-info">
          <span className="tenant-label">Tenant:</span>
          <span className="tenant-name">{tenant?.name}</span>
        </div>
        <p className="settings-description">
          Update company contact details and privacy preferences below.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="settings-form">
        <div className="form-grid">
          {/* Company Name */}
          <div className="form-group">
            <label htmlFor="companyName" className="form-label">
              Company Name
            </label>
            <input
              id="companyName"
              name="companyName"
              type="text"
              className="form-input"
              value={form.companyName}
              onChange={handleChange}
              required
            />
          </div>

          {/* Contact Email */}
          <div className="form-group">
            <label htmlFor="contactEmail" className="form-label">
              Contact Email
            </label>
            <input
              id="contactEmail"
              name="contactEmail"
              type="email"
              className="form-input"
              value={form.contactEmail}
              onChange={handleChange}
              required
            />
          </div>

          {/* Phone Number */}
          <div className="form-group">
            <label htmlFor="phoneNumber" className="form-label">
              Phone Number
            </label>
            <input
              id="phoneNumber"
              name="phoneNumber"
              type="tel"
              className="form-input"
              pattern="^[0-9()+\\-\\s]*$"
              placeholder="+1 (123) 456-7890"
              value={form.phoneNumber}
              onChange={handleChange}
            />
          </div>

          {/* Company Description */}
          <div className="form-group form-group-full">
            <label htmlFor="description" className="form-label">
              Company Description
            </label>
            <textarea
              id="description"
              name="description"
              className="form-textarea"
              rows={3}
              placeholder="Brief description of what your company does"
              value={form.description}
              onChange={handleChange}
            />
          </div>

          {/* Privacy Policy */}
          <div className="form-group form-group-full">
            <label htmlFor="privacyPolicy" className="form-label">
              Privacy Policy
            </label>
            <textarea
              id="privacyPolicy"
              name="privacyPolicy"
              className="form-textarea"
              rows={6}
              placeholder="Add or edit your company's privacy policy here"
              value={form.privacyPolicy}
              onChange={handleChange}
            />
          </div>
        </div>

        <div className="form-actions">
          <button
            type="submit"
            className="submit-button"
            disabled={status === "saving"}
          >
            {status === "saving" ? "Saving..." : "Save Changes"}
          </button>

          {status === "success" && (
            <div className="success-message">
              <span className="success-icon">✓</span>
              <span>Changes saved successfully!</span>
            </div>
          )}
        </div>
      </form>
    </section>
  );
}
