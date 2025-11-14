"""
Carnico Inventory Sync API
FastAPI backend for syncing ERP data to cloud inventory
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, date
import os
from supabase import create_client, Client
import hashlib
import json

# Initialize FastAPI
app = FastAPI(
    title="Carnico Inventory API",
    description="Sync API for Carnico ERP System",
    version="1.0.0"
)

# CORS - Allow requests from anywhere (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase setup
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# API Key for authentication (simple approach)
API_KEY = os.getenv("API_KEY", "carnico_sync_key_2025")  # Change this!

# ============================================================================
# Pydantic Models (Request/Response schemas)
# ============================================================================

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
    customer_no: Optional[int] = None
    specie: Optional[str] = None
    grade: Optional[str] = None
    status: str = "open"
    total_weight: float = 0
    total_amount: float = 0
    labels: List[Label] = []

class ActivityLog(BaseModel):
    action: str
    batch_id: Optional[int] = None
    barcode: Optional[str] = None
    product: Optional[str] = None
    weight: Optional[float] = None
    price: Optional[float] = None
    user: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Optional[dict] = None

class DeletionLog(BaseModel):
    serial_number: str
    product_name: str
    weight: float
    total_price: float
    batch_number: str
    original_timestamp: str
    deleted_at: str
    deleted_by: str

# ============================================================================
# Authentication
# ============================================================================

def verify_api_key(x_api_key: str = Header(...)):
    """Verify API key from header"""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

# ============================================================================
# Helper Functions
# ============================================================================

def calculate_hash(data: dict) -> str:
    """Calculate MD5 hash of data for change detection"""
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()

def needs_sync(entity_type: str, entity_id: str, current_hash: str) -> bool:
    """Check if entity needs syncing based on hash"""
    try:
        result = supabase.table('sync_status').select('sync_hash').eq('entity_type', entity_type).eq('entity_id', entity_id).execute()
        if not result.data:
            return True
        return result.data[0]['sync_hash'] != current_hash
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
        }).execute()
    except Exception as e:
        print(f"Failed to update sync status: {e}")

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
def root():
    """Health check"""
    return {
        "status": "online",
        "service": "Carnico Inventory API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    try:
        # Test database connection
        supabase.table('batches').select('count').limit(1).execute()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "api": "online",
        "database": db_status,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# Batch Endpoints
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
        
        supabase.table('batches').upsert(batch_record).execute()
        
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
            supabase.table('labels').upsert(label_record).execute()
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
        labels_result = supabase.table('labels').select('*').eq('batch_id', batch_id).eq('is_deleted', False).execute()
        
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
# Activity Log Endpoints
# ============================================================================

@app.post("/api/activity/sync", dependencies=[Depends(verify_api_key)])
def sync_activity_logs(logs: List[ActivityLog]):
    """Sync multiple activity log entries"""
    try:
        synced_count = 0
        for log in logs:
            log_record = {
                'action': log.action,
                'batch_id': log.batch_id,
                'barcode': log.barcode,
                'product': log.product,
                'weight': log.weight,
                'price': log.price,
                'user_name': log.user,
                'timestamp': log.timestamp or datetime.now().isoformat(),
                'metadata': log.metadata
            }
            supabase.table('activity_log').insert(log_record).execute()
            synced_count += 1
        
        return {
            "status": "success",
            "message": f"Synced {synced_count} activity logs"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/activity", dependencies=[Depends(verify_api_key)])
def get_activity_logs(batch_id: Optional[int] = None, limit: int = 100):
    """Get activity logs"""
    try:
        query = supabase.table('activity_log').select('*').order('timestamp', desc=True).limit(limit)
        
        if batch_id:
            query = query.eq('batch_id', batch_id)
        
        result = query.execute()
        
        return {
            "status": "success",
            "count": len(result.data),
            "logs": result.data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Deletion Log Endpoints
# ============================================================================

@app.post("/api/deletions/sync", dependencies=[Depends(verify_api_key)])
def sync_deletion_logs(logs: List[DeletionLog]):
    """Sync deletion log entries"""
    try:
        synced_count = 0
        for log in logs:
            log_record = {
                'serial_number': log.serial_number,
                'product_name': log.product_name,
                'weight': log.weight,
                'total_price': log.total_price,
                'batch_number': log.batch_number,
                'original_timestamp': log.original_timestamp,
                'deleted_at': log.deleted_at,
                'deleted_by': log.deleted_by
            }
            supabase.table('deletion_log').insert(log_record).execute()
            
            # Mark label as deleted
            supabase.table('labels').update({
                'is_deleted': True,
                'deleted_at': log.deleted_at
            }).eq('barcode', log.serial_number).execute()
            
            synced_count += 1
        
        return {
            "status": "success",
            "message": f"Synced {synced_count} deletion logs"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Statistics Endpoints
# ============================================================================

@app.get("/api/stats", dependencies=[Depends(verify_api_key)])
def get_statistics():
    """Get inventory statistics"""
    try:
        # Count batches
        total_batches = supabase.table('batches').select('count').execute()
        open_batches = supabase.table('batches').select('count').eq('status', 'open').execute()
        
        # Count labels
        total_labels = supabase.table('labels').select('count').eq('is_deleted', False).execute()
        deleted_labels = supabase.table('labels').select('count').eq('is_deleted', True).execute()
        
        # Sum totals
        totals = supabase.table('batches').select('total_weight, total_amount').execute()
        total_weight = sum(b.get('total_weight', 0) for b in totals.data)
        total_amount = sum(b.get('total_amount', 0) for b in totals.data)
        
        return {
            "status": "success",
            "statistics": {
                "total_batches": len(total_batches.data) if total_batches.data else 0,
                "open_batches": len(open_batches.data) if open_batches.data else 0,
                "total_labels": len(total_labels.data) if total_labels.data else 0,
                "deleted_labels": len(deleted_labels.data) if deleted_labels.data else 0,
                "total_weight_kg": round(total_weight, 3),
                "total_value": round(total_amount, 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
