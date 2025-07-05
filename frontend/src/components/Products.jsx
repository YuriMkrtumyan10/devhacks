import React, { useState, useEffect } from "react";
import { productsApi } from "../services/api";
import { useTenant } from "./TenantContext";

export default function Products() {
  const { tenant } = useTenant();
  const [items, setItems] = useState({ products: [], offerings: [], total: 0 });
  const [loading, setLoading] = useState(true);
  const [operationLoading, setOperationLoading] = useState(false);
  const [error, setError] = useState("");
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [itemToDelete, setItemToDelete] = useState(null);
  const [editingItem, setEditingItem] = useState(null);
  const [activeTab, setActiveTab] = useState("all");
  const [newItem, setNewItem] = useState({
    name: "",
    description: "",
    price: "",
    type: "product",
    stock_qty: "",
    currency: "USD",
  });

  useEffect(() => {
    fetchItems();
  }, [tenant]);

  const fetchItems = async () => {
    if (!tenant?.id) return;

    setLoading(true);
    setError("");
    try {
      const data = await productsApi.getCompanyItems(tenant.id);
      setItems(data);
    } catch (err) {
      setError(err.message);
      setItems({ products: [], offerings: [], total: 0 });
    } finally {
      setLoading(false);
    }
  };

  const handleCreateItem = async (e) => {
    e.preventDefault();
    setOperationLoading(true);
    setError("");

    // Validate tenant is available
    if (!tenant?.id) {
      setError("Company information is not available. Please try again.");
      setOperationLoading(false);
      return;
    }

    try {
      const itemData = {
        ...newItem,
        company_id: tenant.id,
        price: parseFloat(newItem.price) || null,
        stock_qty:
          newItem.type === "product"
            ? parseInt(newItem.stock_qty) || 0
            : undefined,
      };

      if (newItem.type === "product") {
        await productsApi.createProduct(itemData);
      } else {
        await productsApi.createOffering({
          ...itemData,
          type: newItem.type === "service" ? "service" : "product",
        });
      }

      setIsCreateModalOpen(false);
      resetForm();
      await fetchItems();
    } catch (err) {
      setError(`Failed to create ${newItem.type}: ${err.message}`);
    } finally {
      setOperationLoading(false);
    }
  };

  const handleEditItem = (item) => {
    setEditingItem({
      ...item,
      price: item.price?.toString() || "",
      stock_qty: item.stock_qty?.toString() || "",
    });
    setIsEditModalOpen(true);
  };

  const handleUpdateItem = async () => {
    setOperationLoading(true);
    setError("");

    // Validate required data
    if (!editingItem?.id) {
      setError("Item information is missing. Please try again.");
      setOperationLoading(false);
      return;
    }

    try {
      const itemData = {
        name: editingItem.name,
        description: editingItem.description,
        price: parseFloat(editingItem.price) || null,
      };

      if (editingItem.type === "product") {
        itemData.stock_qty = parseInt(editingItem.stock_qty) || 0;
        await productsApi.updateProduct(editingItem.id, itemData);
      } else {
        itemData.currency = editingItem.currency;
        await productsApi.updateOffering(editingItem.id, itemData);
      }

      setIsEditModalOpen(false);
      setEditingItem(null);
      await fetchItems();
    } catch (err) {
      setError(`Failed to update ${editingItem.type}: ${err.message}`);
    } finally {
      setOperationLoading(false);
    }
  };

  const handleDeleteClick = (item) => {
    setItemToDelete(item);
    setIsDeleteModalOpen(true);
  };

  const handleConfirmDelete = async () => {
    if (!itemToDelete) return;

    setOperationLoading(true);
    setError("");
    try {
      if (itemToDelete.type === "product") {
        await productsApi.deleteProduct(itemToDelete.id);
      } else {
        await productsApi.deleteOffering(itemToDelete.id);
      }

      setIsDeleteModalOpen(false);
      setItemToDelete(null);
      await fetchItems();
    } catch (err) {
      setError(`Failed to delete ${itemToDelete.type}: ${err.message}`);
    } finally {
      setOperationLoading(false);
    }
  };

  const resetForm = () => {
    setNewItem({
      name: "",
      description: "",
      price: "",
      type: "product",
      stock_qty: "",
      currency: "USD",
    });
  };

  const getFilteredItems = () => {
    const allItems = [
      ...items.products.map((p) => ({ ...p, category: "Product" })),
      ...items.offerings.map((o) => ({
        ...o,
        category: o.type === "service" ? "Service" : "Product",
      })),
    ];

    switch (activeTab) {
      case "products":
        return items.products.map((p) => ({ ...p, category: "Product" }));
      case "services":
        return items.offerings
          .filter((o) => o.type === "service")
          .map((o) => ({ ...o, category: "Service" }));
      case "all":
      default:
        return allItems;
    }
  };

  const filteredItems = getFilteredItems();

  if (loading && items.total === 0) {
    return (
      <div className="cosmic-loading">
        <div className="cosmic-spinner-large"></div>
        <h2 className="text-xl lg:text-2xl font-semibold">
          Loading products & services...
        </h2>
        <p className="text-base lg:text-lg">Fetching your company offerings</p>
      </div>
    );
  }

  return (
    <div className="cosmic-products">
      <div className="products-header">
        <div className="header-content">
          <h1 className="text-3xl lg:text-4xl font-bold leading-tight">
            📦 Products & Services
          </h1>
          <p className="text-base lg:text-lg leading-relaxed">
            Manage your company's products and service offerings
          </p>
          {tenant && (
            <div className="tenant-info">
              <span className="tenant-label">Company:</span>
              <span className="tenant-name">{tenant.name}</span>
            </div>
          )}
        </div>

        <div className="header-actions">
          <button
            className="cosmic-btn primary"
            onClick={() => setIsCreateModalOpen(true)}
          >
            <span className="btn-icon">✨</span>
            <span className="hidden sm:inline">Add Product/Service</span>
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

      {/* Stats */}
      <div className="products-stats grid-cols-3 lg:grid-cols-3">
        <div className="stat-card">
          <div className="stat-icon">📦</div>
          <div className="stat-info">
            <h3 className="text-xl lg:text-2xl font-bold">
              {items.products.length}
            </h3>
            <p className="text-sm lg:text-base">Products</p>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">🛠️</div>
          <div className="stat-info">
            <h3 className="text-xl lg:text-2xl font-bold">
              {items.offerings.filter((o) => o.type === "service").length}
            </h3>
            <p className="text-sm lg:text-base">Services</p>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">💰</div>
          <div className="stat-info">
            <h3 className="text-xl lg:text-2xl font-bold">
              $
              {[...items.products, ...items.offerings]
                .reduce((sum, item) => sum + (item.price || 0), 0)
                .toFixed(0)}
            </h3>
            <p className="text-sm lg:text-base">Total Value</p>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="cosmic-tabs">
        <button
          className={`tab-btn ${activeTab === "all" ? "active" : ""}`}
          onClick={() => setActiveTab("all")}
        >
          All Items ({items.total})
        </button>
        <button
          className={`tab-btn ${activeTab === "products" ? "active" : ""}`}
          onClick={() => setActiveTab("products")}
        >
          Products ({items.products.length})
        </button>
        <button
          className={`tab-btn ${activeTab === "services" ? "active" : ""}`}
          onClick={() => setActiveTab("services")}
        >
          Services ({items.offerings.filter((o) => o.type === "service").length}
          )
        </button>
      </div>

      {/* Items Grid */}
      {filteredItems.length > 0 ? (
        <div className="products-grid">
          {filteredItems.map((item) => (
            <div key={`${item.type}-${item.id}`} className="product-card">
              <div className="product-header">
                <div className="product-icon">
                  {item.type === "product" ? "📦" : "🛠️"}
                </div>
                <div className="product-info">
                  <h3 className="product-name text-lg font-bold">
                    {item.name}
                  </h3>
                  <span className="product-category text-sm">
                    {item.category}
                  </span>
                </div>
                <div className="product-actions">
                  <button
                    className="action-btn edit"
                    onClick={() => handleEditItem(item)}
                    title="Edit Item"
                    disabled={operationLoading}
                  >
                    ✏️
                  </button>
                  <button
                    className="action-btn delete"
                    onClick={() => handleDeleteClick(item)}
                    title="Delete Item"
                    disabled={operationLoading}
                  >
                    🗑️
                  </button>
                </div>
              </div>

              <div className="product-details">
                {item.description && (
                  <p className="product-description text-sm">
                    {item.description}
                  </p>
                )}

                <div className="detail-row">
                  <span className="detail-label text-xs">Price:</span>
                  <span className="detail-value text-xs font-bold">
                    ${item.price || "0.00"}
                    {item.currency &&
                      item.currency !== "USD" &&
                      ` ${item.currency}`}
                  </span>
                </div>

                {item.type === "product" && (
                  <div className="detail-row">
                    <span className="detail-label text-xs">Stock:</span>
                    <span className="detail-value text-xs">
                      {item.stock_qty || 0} units
                    </span>
                  </div>
                )}

                <div className="detail-row">
                  <span className="detail-label text-xs">Created:</span>
                  <span className="detail-value text-xs">
                    {item.created_at
                      ? new Date(item.created_at).toLocaleDateString()
                      : "Unknown"}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="cosmic-empty-state">
          <div className="empty-icon">📦</div>
          <h3 className="text-xl lg:text-2xl font-semibold">No items found</h3>
          <p className="text-base lg:text-lg">
            {activeTab === "all"
              ? "No products or services have been added yet."
              : `No ${activeTab} found.`}{" "}
            Create your first item to get started.
          </p>
          <button
            className="cosmic-btn primary"
            onClick={() => setIsCreateModalOpen(true)}
          >
            <span className="btn-icon">✨</span>
            Add {activeTab === "services" ? "Service" : "Product"}
          </button>
        </div>
      )}

      {/* Create Modal */}
      {isCreateModalOpen && (
        <div className="cosmic-modal">
          <div
            className="modal-backdrop"
            onClick={() => setIsCreateModalOpen(false)}
          ></div>
          <div className="modal-content">
            <div className="modal-header">
              <h2 className="text-xl lg:text-2xl font-bold">
                ✨ Create New{" "}
                {newItem.type === "product" ? "Product" : "Service"}
              </h2>
              <button
                className="close-btn"
                onClick={() => setIsCreateModalOpen(false)}
              >
                ✕
              </button>
            </div>

            {!tenant?.id && (
              <div
                className="cosmic-alert warning"
                style={{ marginBottom: "1rem" }}
              >
                <span className="alert-icon">⚠️</span>
                <div className="alert-content">
                  <h4 className="font-semibold">Company Required</h4>
                  <p className="text-sm">
                    Please select a company to create products or services.
                  </p>
                </div>
              </div>
            )}

            <form onSubmit={handleCreateItem} className="product-form">
              <div className="form-grid">
                <div className="form-group">
                  <label htmlFor="type" className="text-sm font-semibold">
                    Type *
                  </label>
                  <select
                    id="type"
                    value={newItem.type}
                    onChange={(e) =>
                      setNewItem({ ...newItem, type: e.target.value })
                    }
                    className="text-base"
                  >
                    <option value="product">Product</option>
                    <option value="service">Service</option>
                  </select>
                </div>

                <div className="form-group">
                  <label htmlFor="name" className="text-sm font-semibold">
                    Name *
                  </label>
                  <input
                    id="name"
                    type="text"
                    value={newItem.name}
                    onChange={(e) =>
                      setNewItem({ ...newItem, name: e.target.value })
                    }
                    required
                    placeholder="Enter item name"
                    className="text-base"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="price" className="text-sm font-semibold">
                    Price
                  </label>
                  <input
                    id="price"
                    type="number"
                    min="0"
                    step="0.01"
                    value={newItem.price}
                    onChange={(e) =>
                      setNewItem({ ...newItem, price: e.target.value })
                    }
                    placeholder="0.00"
                    className="text-base"
                  />
                </div>

                {newItem.type === "product" && (
                  <div className="form-group">
                    <label
                      htmlFor="stock_qty"
                      className="text-sm font-semibold"
                    >
                      Stock Quantity
                    </label>
                    <input
                      id="stock_qty"
                      type="number"
                      min="0"
                      value={newItem.stock_qty}
                      onChange={(e) =>
                        setNewItem({ ...newItem, stock_qty: e.target.value })
                      }
                      placeholder="0"
                      className="text-base"
                    />
                  </div>
                )}

                {newItem.type === "service" && (
                  <div className="form-group">
                    <label htmlFor="currency" className="text-sm font-semibold">
                      Currency
                    </label>
                    <select
                      id="currency"
                      value={newItem.currency}
                      onChange={(e) =>
                        setNewItem({ ...newItem, currency: e.target.value })
                      }
                      className="text-base"
                    >
                      <option value="USD">USD</option>
                      <option value="EUR">EUR</option>
                      <option value="GBP">GBP</option>
                      <option value="CAD">CAD</option>
                    </select>
                  </div>
                )}
              </div>

              <div className="form-group full-width">
                <label htmlFor="description" className="text-sm font-semibold">
                  Description
                </label>
                <textarea
                  id="description"
                  value={newItem.description}
                  onChange={(e) =>
                    setNewItem({ ...newItem, description: e.target.value })
                  }
                  placeholder="Describe your product or service"
                  className="text-base"
                  rows="3"
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
                  disabled={
                    operationLoading || !newItem.name.trim() || !tenant?.id
                  }
                >
                  {operationLoading ? (
                    <>
                      <div className="cosmic-spinner"></div>
                      Creating...
                    </>
                  ) : (
                    <>
                      <span className="btn-icon">✨</span>
                      Create{" "}
                      {newItem.type === "product" ? "Product" : "Service"}
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Edit Modal */}
      {isEditModalOpen && editingItem && (
        <div className="cosmic-modal">
          <div
            className="modal-backdrop"
            onClick={() => setIsEditModalOpen(false)}
          ></div>
          <div className="modal-content">
            <div className="modal-header">
              <h2 className="text-xl lg:text-2xl font-bold">
                ✏️ Edit {editingItem.type === "product" ? "Product" : "Service"}
              </h2>
              <button
                className="close-btn"
                onClick={() => setIsEditModalOpen(false)}
              >
                ✕
              </button>
            </div>

            <form className="product-form">
              <div className="form-grid">
                <div className="form-group">
                  <label htmlFor="edit-name" className="text-sm font-semibold">
                    Name *
                  </label>
                  <input
                    id="edit-name"
                    type="text"
                    value={editingItem.name}
                    onChange={(e) =>
                      setEditingItem({ ...editingItem, name: e.target.value })
                    }
                    required
                    className="text-base"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="edit-price" className="text-sm font-semibold">
                    Price
                  </label>
                  <input
                    id="edit-price"
                    type="number"
                    min="0"
                    step="0.01"
                    value={editingItem.price}
                    onChange={(e) =>
                      setEditingItem({ ...editingItem, price: e.target.value })
                    }
                    className="text-base"
                  />
                </div>

                {editingItem.type === "product" && (
                  <div className="form-group">
                    <label
                      htmlFor="edit-stock_qty"
                      className="text-sm font-semibold"
                    >
                      Stock Quantity
                    </label>
                    <input
                      id="edit-stock_qty"
                      type="number"
                      min="0"
                      value={editingItem.stock_qty}
                      onChange={(e) =>
                        setEditingItem({
                          ...editingItem,
                          stock_qty: e.target.value,
                        })
                      }
                      className="text-base"
                    />
                  </div>
                )}

                {editingItem.type === "service" && (
                  <div className="form-group">
                    <label
                      htmlFor="edit-currency"
                      className="text-sm font-semibold"
                    >
                      Currency
                    </label>
                    <select
                      id="edit-currency"
                      value={editingItem.currency || "USD"}
                      onChange={(e) =>
                        setEditingItem({
                          ...editingItem,
                          currency: e.target.value,
                        })
                      }
                      className="text-base"
                    >
                      <option value="USD">USD</option>
                      <option value="EUR">EUR</option>
                      <option value="GBP">GBP</option>
                      <option value="CAD">CAD</option>
                    </select>
                  </div>
                )}
              </div>

              <div className="form-group full-width">
                <label
                  htmlFor="edit-description"
                  className="text-sm font-semibold"
                >
                  Description
                </label>
                <textarea
                  id="edit-description"
                  value={editingItem.description || ""}
                  onChange={(e) =>
                    setEditingItem({
                      ...editingItem,
                      description: e.target.value,
                    })
                  }
                  className="text-base"
                  rows="3"
                />
              </div>
            </form>

            <div className="modal-actions">
              <button
                type="button"
                className="cosmic-btn secondary"
                onClick={() => setIsEditModalOpen(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="cosmic-btn primary"
                onClick={handleUpdateItem}
                disabled={operationLoading || !editingItem.name.trim()}
              >
                {operationLoading ? (
                  <>
                    <div className="cosmic-spinner"></div>
                    Updating...
                  </>
                ) : (
                  <>
                    <span className="btn-icon">✏️</span>
                    Update{" "}
                    {editingItem.type === "product" ? "Product" : "Service"}
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {isDeleteModalOpen && itemToDelete && (
        <div className="cosmic-modal">
          <div
            className="modal-backdrop"
            onClick={() => setIsDeleteModalOpen(false)}
          ></div>
          <div className="modal-content">
            <div className="modal-header">
              <h2 className="text-xl lg:text-2xl font-bold">
                🗑️ Delete{" "}
                {itemToDelete.type === "product" ? "Product" : "Service"}
              </h2>
              <button
                className="close-btn"
                onClick={() => setIsDeleteModalOpen(false)}
              >
                ✕
              </button>
            </div>

            <div className="modal-body">
              <p className="text-base lg:text-lg mb-4">
                Are you sure you want to delete{" "}
                <strong>{itemToDelete.name}</strong>?
              </p>
              <p className="text-sm text-gray-600">
                This action cannot be undone. The {itemToDelete.type} will be
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
                    Delete{" "}
                    {itemToDelete.type === "product" ? "Product" : "Service"}
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
