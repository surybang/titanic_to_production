#bin/bash

python3 docs/train.py
uvicorn app.api:app --host "0.0.0.0"