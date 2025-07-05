from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import Optional
from datetime import datetime, timedelta

from models import Company, User, Product, Offering
from database import get_db

router = APIRouter(tags=["dashboard"])

@router.get("/admin")
def get_admin_dashboard_stats(
    period: str = Query("30d", regex="^(7d|30d|90d|1y)$"),
    db: Session = Depends(get_db),
):
    """Get comprehensive admin dashboard statistics"""
    
    # Calculate date range based on period
    now = datetime.utcnow()
    if period == "7d":
        start_date = now - timedelta(days=7)
    elif period == "30d":
        start_date = now - timedelta(days=30)
    elif period == "90d":
        start_date = now - timedelta(days=90)
    else:  # 1y
        start_date = now - timedelta(days=365)
    
    # Total counts
    total_companies = db.query(Company).count()
    total_users = db.query(User).count()
    total_products = db.query(Product).count()
    total_offerings = db.query(Offering).count()
    
    # New counts in period
    new_companies = db.query(Company).filter(Company.created_at >= start_date).count()
    new_users = db.query(User).filter(User.created_at >= start_date).count()
    new_products = db.query(Product).filter(Product.created_at >= start_date).count()
    new_offerings = db.query(Offering).filter(Offering.created_at >= start_date).count()
    
    # Countries count
    active_countries = db.query(func.count(func.distinct(Company.country))).filter(
        Company.country.isnot(None),
        Company.country != ''
    ).scalar() or 0
    
    # Industry distribution
    industry_stats = db.query(
        Company.industry,
        func.count(Company.id).label('count')
    ).filter(
        Company.industry.isnot(None),
        Company.industry != ''
    ).group_by(Company.industry).all()
    
    top_industries = [
        {
            "name": industry,
            "count": count,
            "growth": round((count / total_companies) * 100, 1) if total_companies > 0 else 0
        }
        for industry, count in industry_stats[:5]
    ]
    
    # Revenue estimation (mock data for now)
    total_revenue = total_companies * 50000  # Estimate
    revenue_growth = 15.3
    
    # Monthly trend data (mock for now)
    revenue_chart = []
    for i in range(6):
        month_date = now - timedelta(days=30 * (5 - i))
        month_name = month_date.strftime("%b")
        revenue_chart.append({
            "month": month_name,
            "revenue": 145000 + (i * 15000) + (new_companies * 1000),
            "companies": max(1, total_companies - (5 - i) * 2)
        })
    
    return {
        "totalCompanies": total_companies,
        "newCompanies": new_companies,
        "totalUsers": total_users,
        "newUsers": new_users,
        "totalProducts": total_products + total_offerings,
        "newProducts": new_products + new_offerings,
        "totalRevenue": total_revenue,
        "revenueGrowth": revenue_growth,
        "activeCountries": active_countries,
        "topIndustries": top_industries,
        "revenueChart": revenue_chart,
        "growthMetrics": {
            "userGrowthRate": round((new_users / max(1, total_users)) * 100, 1),
            "companyGrowthRate": round((new_companies / max(1, total_companies)) * 100, 1),
            "revenueGrowthRate": revenue_growth,
            "productGrowthRate": round(((new_products + new_offerings) / max(1, total_products + total_offerings)) * 100, 1)
        }
    }

@router.get("/company/{company_id}")
def get_company_dashboard_stats(
    company_id: int,
    period: str = Query("30d", regex="^(7d|30d|90d)$"),
    db: Session = Depends(get_db),
):
    """Get company-specific dashboard statistics"""
    
    # Verify company exists
    company = db.get(Company, company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    # Calculate date range
    now = datetime.utcnow()
    if period == "7d":
        start_date = now - timedelta(days=7)
    elif period == "30d":
        start_date = now - timedelta(days=30)
    else:  # 90d
        start_date = now - timedelta(days=90)
    
    # Company products and offerings
    total_products = db.query(Product).filter(Product.company_id == company_id).count()
    total_offerings = db.query(Offering).filter(Offering.company_id == company_id).count()
    
    new_products = db.query(Product).filter(
        Product.company_id == company_id,
        Product.created_at >= start_date
    ).count()
    
    new_offerings = db.query(Offering).filter(
        Offering.company_id == company_id,
        Offering.created_at >= start_date
    ).count()
    
    # Team members (users in this company)
    team_members = db.query(User).filter(User.company_id == company_id).count()
    
    # Revenue estimation based on products/offerings
    estimated_revenue = (total_products * 5000) + (total_offerings * 3000)
    revenue_growth = 12.5
    
    # Mock order data
    orders = 147
    new_orders = 18
    customers = 342
    new_customers = 27
    
    # Monthly revenue trend
    revenue_chart = []
    base_revenue = estimated_revenue // 6
    for i in range(6):
        month_date = now - timedelta(days=30 * (5 - i))
        month_name = month_date.strftime("%b")
        revenue_chart.append({
            "month": month_name,
            "revenue": base_revenue + (i * 1000) + (new_products * 500),
            "orders": max(20, orders - (5 - i) * 10)
        })
    
    # Product performance (mock data)
    product_performance = [
        {"name": "Premium Plan", "revenue": estimated_revenue * 0.4, "growth": 15.2},
        {"name": "Basic Plan", "revenue": estimated_revenue * 0.3, "growth": 8.7},
        {"name": "Enterprise", "revenue": estimated_revenue * 0.3, "growth": 22.3}
    ]
    
    return {
        "totalProducts": total_products + total_offerings,
        "newProducts": new_products + new_offerings,
        "revenue": estimated_revenue,
        "revenueGrowth": revenue_growth,
        "orders": orders,
        "newOrders": new_orders,
        "teamMembers": team_members,
        "customers": customers,
        "newCustomers": new_customers,
        "conversionRate": 3.4,
        "avgOrderValue": 156.80,
        "revenueChart": revenue_chart,
        "productPerformance": product_performance,
        "customerMetrics": {
            "retention": 89.5,
            "satisfaction": 4.7,
            "churnRate": 2.3,
            "lifetimeValue": 1240
        }
    }
