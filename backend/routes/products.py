from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from decimal import Decimal

from models import Product, Offering, OfferingType
from database import get_db

router = APIRouter(tags=["products"])

# Pydantic schemas
class ProductBase(BaseModel):
    name: str
    description: Optional[str] = None
    price: Optional[Decimal] = None
    stock_qty: Optional[int] = 0

class ProductCreate(ProductBase):
    company_id: int

class ProductUpdate(ProductBase):
    pass

class ProductResponse(ProductBase):
    id: int
    company_id: int
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True

class OfferingBase(BaseModel):
    name: str
    type: OfferingType
    description: Optional[str] = None
    price: Optional[Decimal] = None
    currency: Optional[str] = "USD"

class OfferingCreate(OfferingBase):
    company_id: int

class OfferingUpdate(OfferingBase):
    pass

class OfferingResponse(OfferingBase):
    id: int
    company_id: int
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True

# Product routes
@router.get("/products", response_model=List[ProductResponse])
def get_company_products(
    company_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """Get all products for a specific company"""
    products = db.query(Product).filter(
        Product.company_id == company_id
    ).offset(skip).limit(limit).all()
    return products

@router.get("/products/{product_id}", response_model=ProductResponse)
def get_product(product_id: int, db: Session = Depends(get_db)):
    product = db.get(Product, product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return product

@router.post("/products", response_model=ProductResponse, status_code=201)
def create_product(product: ProductCreate, db: Session = Depends(get_db)):
    db_product = Product(**product.dict())
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return db_product

@router.put("/products/{product_id}", response_model=ProductResponse)
def update_product(
    product_id: int,
    product: ProductUpdate,
    db: Session = Depends(get_db),
):
    db_product = db.get(Product, product_id)
    if db_product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    
    for field, value in product.dict(exclude_unset=True).items():
        setattr(db_product, field, value)
    
    db.commit()
    db.refresh(db_product)
    return db_product

@router.delete("/products/{product_id}", status_code=204)
def delete_product(product_id: int, db: Session = Depends(get_db)):
    db_product = db.get(Product, product_id)
    if db_product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    
    db.delete(db_product)
    db.commit()

# Offering routes
@router.get("/offerings", response_model=List[OfferingResponse])
def get_company_offerings(
    company_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """Get all offerings for a specific company"""
    offerings = db.query(Offering).filter(
        Offering.company_id == company_id
    ).offset(skip).limit(limit).all()
    return offerings

@router.get("/offerings/{offering_id}", response_model=OfferingResponse)
def get_offering(offering_id: int, db: Session = Depends(get_db)):
    offering = db.get(Offering, offering_id)
    if offering is None:
        raise HTTPException(status_code=404, detail="Offering not found")
    return offering

@router.post("/offerings", response_model=OfferingResponse, status_code=201)
def create_offering(offering: OfferingCreate, db: Session = Depends(get_db)):
    db_offering = Offering(**offering.dict())
    db.add(db_offering)
    db.commit()
    db.refresh(db_offering)
    return db_offering

@router.put("/offerings/{offering_id}", response_model=OfferingResponse)
def update_offering(
    offering_id: int,
    offering: OfferingUpdate,
    db: Session = Depends(get_db),
):
    db_offering = db.get(Offering, offering_id)
    if db_offering is None:
        raise HTTPException(status_code=404, detail="Offering not found")
    
    for field, value in offering.dict(exclude_unset=True).items():
        setattr(db_offering, field, value)
    
    db.commit()
    db.refresh(db_offering)
    return db_offering

@router.delete("/offerings/{offering_id}", status_code=204)
def delete_offering(offering_id: int, db: Session = Depends(get_db)):
    db_offering = db.get(Offering, offering_id)
    if db_offering is None:
        raise HTTPException(status_code=404, detail="Offering not found")
    
    db.delete(db_offering)
    db.commit()

# Combined endpoint for all products and offerings
@router.get("/company/{company_id}/all-items")
def get_all_company_items(company_id: int, db: Session = Depends(get_db)):
    """Get all products and offerings for a company in a combined format"""
    products = db.query(Product).filter(Product.company_id == company_id).all()
    offerings = db.query(Offering).filter(Offering.company_id == company_id).all()
    
    # Format products
    formatted_products = []
    for product in products:
        formatted_products.append({
            "id": product.id,
            "name": product.name,
            "description": product.description,
            "price": float(product.price) if product.price else 0,
            "type": "product",
            "stock_qty": product.stock_qty,
            "created_at": product.created_at.isoformat() if product.created_at else None,
        })
    
    # Format offerings
    formatted_offerings = []
    for offering in offerings:
        formatted_offerings.append({
            "id": offering.id,
            "name": offering.name,
            "description": offering.description,
            "price": float(offering.price) if offering.price else 0,
            "type": offering.type.value,
            "currency": offering.currency,
            "created_at": offering.created_at.isoformat() if offering.created_at else None,
        })
    
    return {
        "products": formatted_products,
        "offerings": formatted_offerings,
        "total": len(formatted_products) + len(formatted_offerings)
    }
