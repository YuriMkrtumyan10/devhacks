#!/usr/bin/env python3
"""
Script to seed the database with initial users for testing
"""
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from database import get_db, engine
from models import User, Company
import bcrypt

def hash_password(password: str) -> str:
    """Hash a password"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def seed_users():
    """Create initial users and companies for testing"""
    
    db = next(get_db())
    
    try:
        # Check if users already exist
        existing_admin = db.query(User).filter(User.email == "admin@saas.com").first()
        if existing_admin:
            print("Users already exist, skipping seed...")
            return
        
        # Create admin company
        admin_company = Company(
            name="Admin Company",
            industry="Technology",
            country="United States",
            contact_email="admin@saas.com"
        )
        db.add(admin_company)
        db.flush()
        
        # Create admin user
        admin_user = User(
            email="admin@saas.com",
            password_hash=hash_password("admin123"),
            role="admin",
            company_id=admin_company.id
        )
        db.add(admin_user)
        
        # Create test companies
        acme_company = Company(
            name="Acme Corp",
            industry="Manufacturing",
            country="United States",
            city="New York",
            contact_email="contact@acme.com",
            website="https://acme.com",
            tagline="We make everything",
            founded_year=1985,
            size="501-1000"
        )
        db.add(acme_company)
        db.flush()
        
        globex_company = Company(
            name="Globex Corporation",
            industry="Technology",
            country="Canada",
            city="Toronto", 
            contact_email="info@globex.com",
            website="https://globex.com",
            tagline="Global solutions for tomorrow",
            founded_year=1995,
            size="201-500"
        )
        db.add(globex_company)
        db.flush()
        
        # Create company owner users
        acme_owner = User(
            email="owner@acme.com",
            password_hash=hash_password("acme123"),
            role="owner",
            company_id=acme_company.id
        )
        db.add(acme_owner)
        
        globex_owner = User(
            email="owner@globex.com", 
            password_hash=hash_password("globex123"),
            role="owner",
            company_id=globex_company.id
        )
        db.add(globex_owner)
        
        # Create a regular user
        test_user = User(
            email="test@acme.com",
            password_hash=hash_password("test123"),
            role="user",
            company_id=acme_company.id
        )
        db.add(test_user)
        
        db.commit()
        
        print("✅ Successfully seeded database with users:")
        print("📧 admin@saas.com (password: admin123) - Admin")
        print("📧 owner@acme.com (password: acme123) - Company Owner")
        print("📧 owner@globex.com (password: globex123) - Company Owner") 
        print("📧 test@acme.com (password: test123) - Regular User")
        
    except Exception as e:
        print(f"❌ Error seeding database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("🌱 Seeding database with initial users...")
    seed_users()
