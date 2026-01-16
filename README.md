# Embedra Project: Documentation

**Sustainable AI Data Management | The "MP3" for Computer Vision Datasets**

## 1. Overview
The **Embedra System** converts heavy raw datasets into a lightweight "Residue" format (ADRF). It is designed with **Privacy by Default**. Your data stays on your machine unless you explicitly choose to upload it to the Cloud Vault.

---

## 2. Quick Start (CLI)

### Installation

**1. Prerequisites**
*   Python 3.10+
*   (Optional) AWS CLI configured if using Cloud Vault.

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```
*Note: If you have issues with PyTorch on Windows, follow the [official PyTorch guide](https://pytorch.org/).*

**3. Configuration**
The project uses a `.env` file to manage API keys and other secrets. Before running the cloud features, you need to:
1.  **Rename `.env.example` to `.env`**.
2.  **Add your API keys** to the `.env` file. The file includes placeholders for your `PINECONE_API_KEY` and AWS credentials.

**4. Setup**
We provide a unified tool: `Embedra.py`.

### Step 1: Process (Private)
Converts raw images into Vectors/Metadata. **No data leaves your computer.**
```bash
python Embedra.py process_all
```
*   **Input:** `data/raw_data/` folder (images) and `data/raw_docs/` (documents).
*   **Output:** `data/adrf/` (The Residue).
*   **Action:** You can now delete the raw images if you only need local metadata search.

**Automatic Curation:**
The system automatically analyzes and sorts your images into four folders inside `data/summaries/preview/`:
1.  **Important:** High-value, unique, or high-complexity images.
2.  **Keep:** Standard quality images that should be retained.
3.  **Review:** Borderline images that may require manual inspection.
4.  **Unnecessary:** Low-quality, duplicates, or low-information images.

### Step 2: Upload (Optional Cloud Backup)
**Only run this if you want to use the Embedra Cloud features.**
This uploads your Vectors to **Pinecone** and your Images to **AWS S3**.
```bash
python Embedra.py upload
```
*   **Cost:** Ingest is free. Retrieval is paid.
*   **Benefit:** Zero-risk backup. Global search. Data marketplace ready.

### Step 3: Search
Query your dataset using natural language.

**Text Search (Private):**
```bash
python Embedra.py search "a green tree"
```

**Document Search (Private):**
```bash
python Embedra.py search_docs "security policy"
```

**Multimodal Search (Images + Docs):**
```bash
python Embedra.py search_multimodal "cyber security" --search_images --search_docs
```

**Cloud Search (Remote):**
```bash
python Embedra.py search "a green tree" --mode cloud
```

---

## 3. Supported Formats
The system automatically detects and processes files in `data/raw_data`:
*   **Images:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.avif`, `.tiff`, `.tif`
*   **Documents:** `.pdf`, `.docx`, `.txt`

## 4. Architecture

| Layer | Technology | Privacy | Description |
| :--- | :--- | :--- | :--- |
| **Ingest** | CLIP ViT-B/32 | **Local** | Runs on your CPU/GPU. Generates embeddings. |
| **Index** | Parquet / Pinecone | **Hybrid** | Local Parquet (Default) or Pinecone (Opt-in). |
| **Storage** | HDD / S3 | **Hybrid** | Local Folder (Default) or AWS S3 (Opt-in). |

## 5. Maintenance
To reset your project and delete all generated data (ADRF files, previews, stats):
```bash
python scripts/clean.py
```

## 6. Confidence Scores (Calibration)
*   **Text Search:** Good matches are **> 25%** (0.25).
*   **Image Search:** Good matches are **> 80%** (0.80).

## 7. Business Model (Cloud Tier)
*   **Tier 1 (Local):** Free. You manage your storage.
*   **Tier 2 (Vault):** We store it. You pay to retrieve/download.
*   **Tier 3 (Market):** You earn credits by sharing data; spend credits to get data.