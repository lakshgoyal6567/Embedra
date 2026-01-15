# Embedra (AI Data Residue Format)

## Overview
A system to convert raw datasets into compact, model-aware representations (ADRF) using embeddings, deduplication, and efficient storage.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
### Convert Dataset
```bash
python adrf_convert.py --input_dir path/to/images --output_file output.adrf.parquet
```

### Python API
```python
from adrf_system.src.reader import ADRFReader

reader = ADRFReader("output.adrf.parquet")
results = reader.search_similarity(query_embedding)
```
