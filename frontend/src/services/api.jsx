import axios from "axios";

const API_BASE_URL = "http://localhost:8000/api";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Add token to requests if available
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("auth_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle token expiration
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem("auth_token");
      localStorage.removeItem("user");
      window.location.reload();
    }
    return Promise.reject(error);
  },
);

export const companyApi = {
  // Get all companies
  getCompanies: async (skip = 0, limit = 100) => {
    const response = await api.get(`/companies?skip=${skip}&limit=${limit}`);
    return response.data;
  },

  // Get a single company by ID
  getCompany: async (id) => {
    const response = await api.get(`/companies/${id}`);
    return response.data;
  },

  // Create a new company
  createCompany: async (companyData) => {
    const response = await api.post("/companies", companyData);
    return response.data;
  },

  // Update an existing company
  updateCompany: async (id, companyData) => {
    const response = await api.put(`/companies/${id}`, companyData);
    return response.data;
  },

  // Delete a company
  deleteCompany: async (id) => {
    await api.delete(`/companies/${id}`);
    return true;
  },

  // Search companies
  searchCompanies: async (query) => {
    const response = await api.get(
      `/companies/search?q=${encodeURIComponent(query)}`,
    );
    return response.data;
  },
};

export const chatApi = {
  // Get all messages
  getMessages: async () => {
    try {
      const response = await api.get("/messages");
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.detail || "Failed to fetch messages",
      );
    }
  },

  // Send a new message
  sendMessage: async (messageData) => {
    try {
      const response = await api.post("/messages", messageData);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.detail || "Failed to send message");
    }
  },
};

export const authApi = {
  // Login user
  login: async (email, password) => {
    const response = await api.post("/auth/login", {
      email: email.trim(),
      password: password,
    });
    return {
      token: response.data.access_token,
      user: response.data.user,
    };
  },

  // Register user
  register: async (email, password, companyName) => {
    const response = await api.post("/auth/register", {
      email: email.trim(),
      password: password,
      company_name: companyName.trim(),
    });
    return {
      token: response.data.access_token,
      user: response.data.user,
    };
  },

  // Logout user
  logout: async () => {
    await api.post("/auth/logout");
    localStorage.removeItem("auth_token");
    localStorage.removeItem("user");
  },

  // Verify token
  verifyToken: async () => {
    const response = await api.get("/auth/verify");
    return response.data;
  },
};

export const dashboardApi = {
  // Get admin dashboard stats (all companies)
  getAdminStats: async (timeFilter = "30d") => {
    const response = await api.get(`/dashboard/admin?period=${timeFilter}`);
    return response.data;
  },

  // Get company dashboard stats
  getCompanyStats: async (companyId, timeFilter = "30d") => {
    const response = await api.get(
      `/dashboard/company/${companyId}?period=${timeFilter}`,
    );
    return response.data;
  },
};

export const productsApi = {
  // Get all items (products + offerings) for a company
  getCompanyItems: async (companyId) => {
    const response = await api.get(`/company/${companyId}/all-items`);
    return response.data;
  },

  // Get products for a company
  getCompanyProducts: async (companyId) => {
    const response = await api.get(`/products?company_id=${companyId}`);
    return response.data;
  },

  // Get offerings for a company
  getCompanyOfferings: async (companyId) => {
    const response = await api.get(`/offerings?company_id=${companyId}`);
    return response.data;
  },

  // Create a new product
  createProduct: async (productData) => {
    const response = await api.post("/products", productData);
    return response.data;
  },

  // Create a new offering
  createOffering: async (offeringData) => {
    const response = await api.post("/offerings", offeringData);
    return response.data;
  },

  // Update a product
  updateProduct: async (id, productData) => {
    const response = await api.put(`/products/${id}`, productData);
    return response.data;
  },

  // Update an offering
  updateOffering: async (id, offeringData) => {
    const response = await api.put(`/offerings/${id}`, offeringData);
    return response.data;
  },

  // Delete a product
  deleteProduct: async (id) => {
    await api.delete(`/products/${id}`);
    return true;
  },

  // Delete an offering
  deleteOffering: async (id) => {
    await api.delete(`/offerings/${id}`);
    return true;
  },

  // Get all products (admin only)
  getAllProducts: async () => {
    const response = await api.get("/products");
    return response.data;
  },
};

export default api;
