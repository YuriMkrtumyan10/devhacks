from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.companies import router as companies_router
from routes.assistant import router as assistant_router
from routes.dashboard import router as dashboard_router
from routes.auth import router as auth_router
from routes.products import router as products_router
from database import engine
from models import Base

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Company Admin API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ToDo: secure origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(auth_router, prefix="/api/auth")
app.include_router(companies_router, prefix="/api/companies")
app.include_router(products_router, prefix="/api")
app.include_router(assistant_router, prefix="/api")
app.include_router(dashboard_router, prefix="/api/dashboard")
