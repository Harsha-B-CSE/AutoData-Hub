import json
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.db.models import PipelineConfig

router = APIRouter(prefix="/api/v1/config", tags=["Configuration & Presets"])

# Built-in Default Presets
DEFAULT_PRESETS = [
    {
        "id": "builtin-tabular-standard",
        "name": "Standard Tabular Hygiene & Normalization",
        "engine_type": "tabular",
        "parameters": {
            "cleaning": {
                "remove_duplicates": True,
                "clean_text": True,
                "text_case": "preserve",
                "infer_types": True,
                "handle_outliers": True,
                "outlier_method": "iqr",
                "outlier_action": "clip",
                "outlier_threshold": 1.5
            },
            "imputation": {
                "numerical_strategy": "median",
                "categorical_strategy": "mode",
                "num_constant": 0.0,
                "cat_constant": "MISSING",
                "knn_neighbors": 5
            },
            "scaling": {
                "method": "standard",
                "feature_range": [0, 1]
            },
            "encoding": {
                "method": "onehot",
                "columns": None,
                "exclude_columns": [],
                "drop_first": False,
                "max_categories": 25
            }
        }
    },
    {
        "id": "builtin-tabular-ml-ready",
        "name": "ML-Ready Tabular (Robust + Frequency)",
        "engine_type": "tabular",
        "parameters": {
            "cleaning": {
                "remove_duplicates": True,
                "clean_text": True,
                "text_case": "lower",
                "infer_types": True,
                "handle_outliers": True,
                "outlier_method": "iqr",
                "outlier_action": "clip",
                "outlier_threshold": 2.0
            },
            "imputation": {
                "numerical_strategy": "knn",
                "categorical_strategy": "mode",
                "num_constant": 0.0,
                "cat_constant": "MISSING",
                "knn_neighbors": 5
            },
            "scaling": {
                "method": "robust"
            },
            "encoding": {
                "method": "frequency",
                "max_categories": 50
            }
        }
    },
    {
        "id": "builtin-vision-standard",
        "name": "Adaptive Vision Enhancement (Balanced - Recommended)",
        "engine_type": "vision",
        "parameters": {
            "transform": {
                "resize": True,
                "width": 512,
                "height": 512,
                "preserve_aspect_ratio": True,
                "interpolation": "bilinear"
            },
            "filter": {
                "color_space": "RGB",
                "adaptive": True,
                "adaptive_preset": "balanced",
                "filter_type": "bilateral",
                "contrast_enhancement": "clahe"
            },
            "augmentation": {
                "enabled": False,
                "horizontal_flip": True,
                "vertical_flip": False,
                "color_jitter": False,
                "gaussian_noise": False
            }
        }
    },
    {
        "id": "builtin-vision-natural",
        "name": "Adaptive Vision Enhancement (Natural / High-Fidelity)",
        "engine_type": "vision",
        "parameters": {
            "transform": {
                "resize": True,
                "width": 512,
                "height": 512,
                "preserve_aspect_ratio": True,
                "interpolation": "bilinear"
            },
            "filter": {
                "color_space": "RGB",
                "adaptive": True,
                "adaptive_preset": "natural",
                "filter_type": "bilateral",
                "contrast_enhancement": "clahe"
            },
            "augmentation": {
                "enabled": False,
                "horizontal_flip": True,
                "vertical_flip": False,
                "color_jitter": False,
                "gaussian_noise": False
            }
        }
    },
    {
        "id": "builtin-vision-strong",
        "name": "Adaptive Vision Enhancement (Strong / Night & Noise)",
        "engine_type": "vision",
        "parameters": {
            "transform": {
                "resize": True,
                "width": 512,
                "height": 512,
                "preserve_aspect_ratio": True,
                "interpolation": "bilinear"
            },
            "filter": {
                "color_space": "RGB",
                "adaptive": True,
                "adaptive_preset": "strong",
                "filter_type": "bilateral",
                "contrast_enhancement": "clahe"
            },
            "augmentation": {
                "enabled": False,
                "horizontal_flip": True,
                "vertical_flip": False,
                "color_jitter": False,
                "gaussian_noise": False
            }
        }
    },
    {
        "id": "builtin-vision-augmented",
        "name": "Training Augmentation & Sharpening (256x256)",
        "engine_type": "vision",
        "parameters": {
            "transform": {
                "resize": True,
                "width": 256,
                "height": 256,
                "preserve_aspect_ratio": True,
                "interpolation": "lanczos"
            },
            "filter": {
                "color_space": "RGB",
                "filter_type": "unsharp",
                "contrast_enhancement": "clahe"
            },
            "augmentation": {
                "enabled": True,
                "horizontal_flip": True,
                "vertical_flip": False,
                "color_jitter": True,
                "gaussian_noise": True
            }
        }
    }
]

class CreatePresetRequest(BaseModel):
    name: str
    engine_type: str  # "tabular" or "vision"
    parameters: Dict[str, Any]

@router.get("/presets")
def get_presets(db: Session = Depends(get_db)):
    db_presets = db.query(PipelineConfig).order_by(PipelineConfig.created_at.desc()).all()
    user_presets = [
        {
            "id": p.id,
            "name": p.name,
            "engine_type": p.engine_type,
            "parameters": json.loads(p.parameters_json),
            "is_custom": True,
            "created_at": p.created_at.isoformat(),
        }
        for p in db_presets
    ]
    return {
        "built_in": DEFAULT_PRESETS,
        "custom": user_presets
    }

@router.post("/presets")
def create_preset(payload: CreatePresetRequest, db: Session = Depends(get_db)):
    cfg = PipelineConfig(
        name=payload.name,
        engine_type=payload.engine_type,
        parameters_json=json.dumps(payload.parameters),
    )
    db.add(cfg)
    db.commit()
    db.refresh(cfg)
    return {
        "config_id": cfg.id,
        "name": cfg.name,
        "status": "saved"
    }

@router.get("/presets/{config_id}")
def get_preset_by_id(config_id: str, db: Session = Depends(get_db)):
    # Check built-in
    for b in DEFAULT_PRESETS:
        if b["id"] == config_id:
            return b

    cfg = db.query(PipelineConfig).filter(PipelineConfig.id == config_id).first()
    if not cfg:
        raise HTTPException(status_code=404, detail="Configuration preset not found")

    return {
        "id": cfg.id,
        "name": cfg.name,
        "engine_type": cfg.engine_type,
        "parameters": json.loads(cfg.parameters_json),
        "is_custom": True,
        "created_at": cfg.created_at.isoformat(),
    }
