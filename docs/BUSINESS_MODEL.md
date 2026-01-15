# 💼 project_name Business Model & Architecture

## The Core Concept: "Smart Glacier"
We offer a storage solution that is **cheaper than keeping it locally** but **smarter than a tape drive**.

---

## 1. The Service Tiers

### 🟢 Tier 1: Local / Self-Managed (The "Tool")
*   **Target:** Developers, Privacy-focused users.
*   **Data Flow:** Client runs `adrf_convert.py`. Keeps `parquet` file. Deletes raw images manually.
*   **Storage:** User's own hard drive.
*   **Cost:** Free (Open Source CLI).
*   **Value:** Compression & Local Search.

### 🔵 Tier 2: project_name Cloud (The "Vault")
*   **Target:** Companies wanting backup & audit trails.
*   **Data Flow:**
    *   **Ingest (Free):** Client uploads Vectors + Raw Images.
    *   We store Vectors in **Pinecone** (Hot).
    *   We store Images in **S3 Deep Archive** (Cold/Cheap).
*   **Retrieval (Paid):**
    *   User searches Vectors (Free/Low cost).
    *   User requests *Download* of original image -> **$$ Fee** (e.g., $0.05/GB + Retrieval Fee).
    *   *Mechanism:* System checks payment -> Thaws image -> Generates temporary download link.

### 🟣 Tier 3: Data Augmentation (The "Marketplace")
*   **Target:** AI teams needing *more* data.
*   **Feature:** "Get images with similar vectors."
*   **Data Flow:**
    *   User provides a vector (from their dataset).
    *   System searches our **Global Index** (anonymized data from other users or public datasets).
    *   User pays to license/download *new* images they didn't have before.
*   **Cost:** Per-image licensing fee.

---

## 2. Technical Architecture

### Client Side (CLI)
We need to update the CLI to support the "Cloud" mode.

```bash
# Option A: Local (Current)
python adrf_convert.py --storage local

# Option B: Cloud Upload (Future)
python adrf_convert.py --storage cloud --api_key YOUR_KEY
```

### Server Side (Backend)
1.  **Vector DB (Pinecone):** Stores the "Index".
2.  **Object Storage (AWS S3 / Wasabi):** Stores the "Blobs" (Images).
3.  **API Gateway (FastAPI):**
    *   `/upload`: Accepts stream, saves to S3.
    *   `/search`: Proxies to Pinecone.
    *   `/retrieve`: Checks billing -> Presigns S3 URL.

---

## 3. Revenue Logic
*   **Inbound Data:** Free (Encourages lock-in).
*   **Outbound Data:** Metered (The "Ransom" model, standard in Cloud Storage).
*   **Compute:** Metered (Search queries).
