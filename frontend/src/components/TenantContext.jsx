import React, { createContext, useContext, useState, useEffect } from "react";
import { companyApi } from "../services/api";
import { useAuth } from "../context/AuthContext";

const TenantContext = createContext();

export function TenantProvider({ children }) {
  const { user } = useAuth();
  const [tenants, setTenants] = useState([]);
  const [tenantId, setTenantId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const tenant = tenants.find((t) => t.id === tenantId);

  useEffect(() => {
    async function fetchTenants() {
      if (!user) return;

      setLoading(true);
      setError(null);

      try {
        let companies = [];

        if (user.role === "admin") {
          // Admin can see all companies except admin companies
          const allCompanies = await companyApi.getCompanies();
          companies = allCompanies.filter(
            (company) =>
              company.name !== "Admin Company" &&
              !company.name.toLowerCase().includes("admin"),
          );
        } else if (user.companyId) {
          // Regular users only see their company
          const company = await companyApi.getCompany(user.companyId);
          companies = [company];
        }

        setTenants(companies);

        // Set initial tenant
        if (companies.length > 0) {
          const initialTenant = user.companyId
            ? companies.find((c) => c.id === user.companyId) || companies[0]
            : companies[0];
          setTenantId(initialTenant.id);
        }
      } catch (err) {
        console.error("Failed to fetch companies:", err);
        setError(err.message);
        setTenants([]);
      } finally {
        setLoading(false);
      }
    }

    fetchTenants();
  }, [user]);

  const refreshTenants = async () => {
    if (!user) return;

    try {
      let companies = [];

      if (user.role === "admin") {
        const allCompanies = await companyApi.getCompanies();
        companies = allCompanies.filter(
          (company) =>
            company.name !== "Admin Company" &&
            !company.name.toLowerCase().includes("admin"),
        );
      } else if (user.companyId) {
        const company = await companyApi.getCompany(user.companyId);
        companies = [company];
      }

      setTenants(companies);
    } catch (err) {
      console.error("Failed to refresh companies:", err);
      setError(err.message);
    }
  };

  const value = {
    tenant,
    tenants,
    tenantId,
    setTenantId,
    loading,
    error,
    refreshTenants,
    isAdmin: user?.role === "admin",
  };

  return (
    <TenantContext.Provider value={value}>{children}</TenantContext.Provider>
  );
}

export function useTenant() {
  const context = useContext(TenantContext);
  if (!context) {
    throw new Error("useTenant must be used within a TenantProvider");
  }
  return context;
}
