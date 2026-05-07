# Melodix Backend — Setup Guide

## Folder structure required

melodix_backend/
├── app.py                  ← Flask API (this file)
├── requirements.txt        ← Python dependencies
├── Procfile                ← For Render/Heroku
├── render.yaml             ← Render auto-deploy config
├── README.md               ← This file
└── artefacts/              ← YOUR MODEL FILES GO HERE
    ├── tracks.csv
    ├── svd_model.pkl
    ├── scaler.pkl
    ├── user_factors.npy
    ├── item_factors.npy
    ├── svd_scores.npy
    ├── feature_array.npy
    ├── train_matrix.npz
    └── test_matrix.npz

## Local run

```bash
pip install -r requirements.txt
python app.py
```

API will run at http://localhost:5000

## Test endpoints locally

```
GET http://localhost:5000/
GET http://localhost:5000/stats
GET http://localhost:5000/recommend?user_id=0&model=svd&k=10
GET http://localhost:5000/recommend?user_id=5&model=hybrid&k=10
GET http://localhost:5000/similar?track_id=T00154&k=8
GET http://localhost:5000/mood?mood=happy&k=15
GET http://localhost:5000/search?q=Taylor&limit=10
GET http://localhost:5000/trending?k=20
GET http://localhost:5000/tracks?sort=energy&order=desc&limit=30
```

## Deploy on Render (free)

See DEPLOYMENT_GUIDE for step-by-step instructions.
