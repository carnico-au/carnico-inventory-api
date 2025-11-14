"""
Carnico Inventory Sync API v2 - Two-way Sync Support
Adds Products, Customers, and Settings sync with manual control
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
import hashlib
import json
from supabase import create_client, Client

# ============================================================================
# Configuration
# ============================================================================
app = FastAPI(title="Carnico Inventory Sync API v2", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
API_KEY = os.getenv("API_KEY", "carnico_2025_butchery_xyz789abc")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================================
# Models
# ============================================================================

class Product(BaseModel):
    product_code: str
    ham_code: Optional[str] = None
    name: str
    specie: Optional[str] = None
    grade: Optional[str] = None
    price: float
    expiry_days: int = 7
    is_active: bool = True

class Customer(BaseModel):
    customer_no: str
    customer_name: str
    contact_person: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    notes: Optional[str] = None
    is_active: bool = True

class Setting(BaseModel):
    setting_key: str
    setting_value: str
    setting_type: str  # 'string', 'number', 'boolean', 'json'
    category: str
    description: Optional[str] = None

class Label(BaseModel):
    barcode: str
    product_name: str
    product_code: Optional[str] = None
    ham_code: Optional[str] = None
    weight: float
    price_per_kg: float
    total_price: float
    grade: Optional[str] = None
    packed_date: Optional[str] = None
    expiry_date: Optional[str] = None
    timestamp: str
    reprint_time: Optional[str] = None

class Batch(BaseModel):
    batch_id: int
    date: str
    customer_name: Optional[str] = None
    customer_no: Optional[str] = None
    specie: Optional[str] = None
    grade: Optional[str] = None
    status: str = "Open"
    total_weight: float = 0
    total_amount: float = 0
    labels: List[Label] = []

class ActivityLog(BaseModel):
    action: str
    timestamp: str
    user: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class DeletionLog(BaseModel):
    barcode: str
    product_name: str
    weight: float
    total_price: str
    batch_id: int
    timestamp: str
    deleted_at: str
    deleted_by: str

# ============================================================================
# Authentication
# ============================================================================

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

# ============================================================================
# Utility Functions
# ============================================================================

def calculate_hash(data: dict) -> str:
    """Calculate hash of data for change detection"""
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()

def needs_sync(entity_type: str, entity_id: str, new_hash: str) -> bool:
    """Check if entity needs syncing based on hash"""
    try:
        result = supabase.table('sync_status').select('sync_hash').eq('entity_type', entity_type).eq('entity_id', entity_id).execute()
        if result.data:
            return result.data[0]['sync_hash'] != new_hash
        return True
    except:
        return True

def update_sync_status(entity_type: str, entity_id: str, sync_hash: str, status: str = 'synced', error: str = None):
    """Update sync status"""
    try:
        supabase.table('sync_status').upsert({
            'entity_type': entity_type,
            'entity_id': entity_id,
            'sync_hash': sync_hash,
            'status': status,
            'error_message': error,
            'last_synced_at': datetime.now().isoformat()
        }, on_conflict='entity_type,entity_id').execute()
    except Exception as e:
        print(f"Failed to update sync status: {e}")

# ============================================================================
# Health Check
# ============================================================================

@app.get("/")
def root():
    """Health check"""
    return {
        "status": "ok",
        "service": "Carnico Inventory Sync API v2",
        "version": "2.0.0",
        "features": ["batches", "labels", "products", "customers", "settings", "two-way-sync"]
    }

@app.get("/health")
def health():
    """Detailed health check"""
    try:
        # Test Supabase connection
        supabase.table('batches').select('batch_id').limit(1).execute()
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ============================================================================
# Products Endpoints (Two-way sync)
# ============================================================================

@app.get("/api/products", dependencies=[Depends(verify_api_key)])
def get_products(active_only: bool = True):
    """Get all products"""
    try:
        query = supabase.table('products').select('*').order('name')
        if active_only:
            query = query.eq('is_active', True)
        result = query.execute()
        return {
            "status": "success",
            "count": len(result.data),
            "products": result.data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/products/sync", dependencies=[Depends(verify_api_key)])
def sync_products(products: List[Product]):
    """Sync products from ERP to cloud (push)"""
    try:
        synced_count = 0
        for product in products:
            product_record = {
                'product_code': product.product_code,
                'ham_code': product.ham_code,
                'name': product.name,
                'specie': product.specie,
                'grade': product.grade,
                'price': product.price,
                'expiry_days': product.expiry_days,
                'is_active': product.is_active,
                'last_sync_at': datetime.now().isoformat()
            }
            supabase.table('products').upsert(product_record, on_conflict='product_code').execute()
            synced_count += 1
        
        return {
            "status": "success",
            "message": f"Synced {synced_count} products",
            "count": synced_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

@app.get("/api/products/pull", dependencies=[Depends(verify_api_key)])
def pull_products(since: Optional[str] = None):
    """Pull products from cloud to ERP"""
    try:
        query = supabase.table('products').select('*').order('updated_at', desc=True)
        if since:
            query = query.gt('updated_at', since)
        result = query.execute()
        return {
            "status": "success",
            "count": len(result.data),
            "products": result.data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Customers Endpoints (Two-way sync)
# ============================================================================

@app.get("/api/customers", dependencies=[Depends(verify_api_key)])
def get_customers(active_only: bool = True):
    """Get all customers"""
    try:
        query = supabase.table('customers').select('*').order('customer_name')
        if active_only:
            query = query.eq('is_active', True)
        result = query.execute()
        return {
            "status": "success",
            "count": len(result.data),
            "customers": result.data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/customers/sync", dependencies=[Depends(verify_api_key)])
def sync_customers(customers: List[Customer]):
    """Sync customers from ERP to cloud (push)"""
    try:
        synced_count = 0
        for customer in customers:
            customer_record = {
                'customer_no': customer.customer_no,
                'customer_name': customer.customer_name,
                'contact_person': customer.contact_person,
                'phone': customer.phone,
                'email': customer.email,
                'address': customer.address,
                'notes': customer.notes,
                'is_active': customer.is_active,
                'last_sync_at': datetime.now().isoformat()
            }
            supabase.table('customers').upsert(customer_record, on_conflict='customer_no').execute()
            synced_count += 1
        
        return {
            "status": "success",
            "message": f"Synced {synced_count} customers",
            "count": synced_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

@app.get("/api/customers/pull", dependencies=[Depends(verify_api_key)])
def pull_customers(since: Optional[str] = None):
    """Pull customers from cloud to ERP"""
    try:
        query = supabase.table('customers').select('*').order('updated_at', desc=True)
        if since:
            query = query.gt('updated_at', since)
        result = query.execute()
        return {
            "status": "success",
            "count": len(result.data),
            "customers": result.data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Settings Endpoints (Two-way sync)
# ============================================================================

@app.get("/api/settings", dependencies=[Depends(verify_api_key)])
def get_settings(category: Optional[str] = None):
    """Get all settings or by category"""
    try:
        query = supabase.table('settings').select('*').order('setting_key')
        if category:
            query = query.eq('category', category)
        result = query.execute()
        return {
            "status": "success",
            "count": len(result.data),
            "settings": result.data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/settings/sync", dependencies=[Depends(verify_api_key)])
def sync_settings(settings: List[Setting]):
    """Sync settings from ERP to cloud (push)"""
    try:
        synced_count = 0
        for setting in settings:
            setting_record = {
                'setting_key': setting.setting_key,
                'setting_value': setting.setting_value,
                'setting_type': setting.setting_type,
                'category': setting.category,
                'description': setting.description,
                'last_sync_at': datetime.now().isoformat()
            }
            supabase.table('settings').upsert(setting_record, on_conflict='setting_key').execute()
            synced_count += 1
        
        return {
            "status": "success",
            "message": f"Synced {synced_count} settings",
            "count": synced_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

@app.get("/api/settings/pull", dependencies=[Depends(verify_api_key)])
def pull_settings(since: Optional[str] = None):
    """Pull settings from cloud to ERP"""
    try:
        query = supabase.table('settings').select('*').order('updated_at', desc=True)
        if since:
            query = query.gt('updated_at', since)
        result = query.execute()
        return {
            "status": "success",
            "count": len(result.data),
            "settings": result.data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Batch Endpoints (Automatic sync)
# ============================================================================

@app.post("/api/batches/sync", dependencies=[Depends(verify_api_key)])
def sync_batch(batch: Batch):
    """Sync a batch with all its labels"""
    try:
        batch_data = batch.dict(exclude={'labels'})
        batch_hash = calculate_hash(batch_data)
        
        # Check if sync needed
        if not needs_sync('batch', str(batch.batch_id), batch_hash):
            return {
                "status": "skipped",
                "message": "Batch unchanged, no sync needed",
                "batch_id": batch.batch_id
            }
        
        # Upsert batch
        batch_record = {
            'batch_id': batch.batch_id,
            'date': batch.date,
            'customer_name': batch.customer_name,
            'customer_no': batch.customer_no,
            'specie': batch.specie,
            'grade': batch.grade,
            'status': batch.status,
            'total_weight': batch.total_weight,
            'total_amount': batch.total_amount,
            'label_count': len(batch.labels),
            'last_sync_at': datetime.now().isoformat()
        }
        
        supabase.table('batches').upsert(batch_record, on_conflict='batch_id').execute()
        
        # Sync labels
        labels_synced = 0
        for label in batch.labels:
            label_record = {
                'batch_id': batch.batch_id,
                'barcode': label.barcode,
                'product_name': label.product_name,
                'product_code': label.product_code,
                'ham_code': label.ham_code,
                'weight': label.weight,
                'price_per_kg': label.price_per_kg,
                'total_price': label.total_price,
                'grade': label.grade,
                'packed_date': label.packed_date,
                'expiry_date': label.expiry_date,
                'timestamp': label.timestamp,
                'reprint_time': label.reprint_time
            }
            supabase.table('labels').upsert(label_record, on_conflict='barcode').execute()
            labels_synced += 1
        
        # Update sync status
        update_sync_status('batch', str(batch.batch_id), batch_hash, 'synced')
        
        return {
            "status": "success",
            "message": "Batch synced successfully",
            "batch_id": batch.batch_id,
            "labels_synced": labels_synced
        }
        
    except Exception as e:
        update_sync_status('batch', str(batch.batch_id), batch_hash, 'failed', str(e))
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

@app.get("/api/batches", dependencies=[Depends(verify_api_key)])
def get_batches(status: Optional[str] = None, limit: int = 100, offset: int = 0):
    """Get all batches with optional filtering"""
    try:
        query = supabase.table('batches').select('*').order('date', desc=True).range(offset, offset + limit - 1)
        
        if status:
            query = query.eq('status', status)
        
        result = query.execute()
        
        return {
            "status": "success",
            "count": len(result.data),
            "batches": result.data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/batches/{batch_id}", dependencies=[Depends(verify_api_key)])
def get_batch(batch_id: int):
    """Get a specific batch with all labels"""
    try:
        # Get batch
        batch_result = supabase.table('batches').select('*').eq('batch_id', batch_id).execute()
        if not batch_result.data:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        # Get labels
        labels_result = supabase.table('labels').select('*').eq('batch_id', batch_id).execute()
        
        batch = batch_result.data[0]
        batch['labels'] = labels_result.data
        
        return {
            "status": "success",
            "batch": batch
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Activity Log Endpoints (Automatic sync)
# ============================================================================

@app.post("/api/activity/sync", dependencies=[Depends(verify_api_key)])
def sync_activity_logs(logs: List[ActivityLog]):
    """Sync multiple activity log entries"""
    try:
        synced_count = 0
        for log in logs:
            log_record = {
                'action': log.action,
                'timestamp': log.timestamp,
                'user_name': log.user,
                'details': log.details
            }
            supabase.table('activity_log').insert(log_record).execute()
            synced_count += 1
        
        return {
            "status": "success",
            "message": f"Synced {synced_count} activity logs",
            "count": synced_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/activity", dependencies=[Depends(verify_api_key)])
def get_activity_logs(limit: int = 100):
    """Get activity logs"""
    try:
        result = supabase.table('activity_log').select('*').order('timestamp', desc=True).limit(limit).execute()
        
        return {
            "status": "success",
            "count": len(result.data),
            "logs": result.data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Deletion Log Endpoints (Automatic sync)
# ============================================================================

@app.post("/api/deletions/sync", dependencies=[Depends(verify_api_key)])
def sync_deletion_logs(logs: List[DeletionLog]):
    """Sync deletion log entries"""
    try:
        synced_count = 0
        for log in logs:
            log_record = {
                'barcode': log.barcode,
                'product_name': log.product_name,
                'weight': log.weight,
                'total_price': log.total_price,
                'batch_id': log.batch_id,
                'timestamp': log.timestamp,
                'deleted_at': log.deleted_at,
                'deleted_by': log.deleted_by
            }
            supabase.table('deletion_log').insert(log_record).execute()
            synced_count += 1
        
        return {
            "status": "success",
            "message": f"Synced {synced_count} deletion logs",
            "count": synced_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/deletions", dependencies=[Depends(verify_api_key)])
def get_deletion_logs(limit: int = 100):
    """Get deletion logs"""
    try:
        result = supabase.table('deletion_log').select('*').order('deleted_at', desc=True).limit(limit).execute()
        
        return {
            "status": "success",
            "count": len(result.data),
            "logs": result.data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
