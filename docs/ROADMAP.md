# 🚀 project_name Product Roadmap (From Prototype to SaaS)

Currently, the system is a **Functional Prototype**. The CLI connects directly to Cloud resources using Admin keys. To launch as a **Commercial SaaS**, the following architecture changes are required.

## 1. 🛡️ The Security Layer (API Gateway)
*   **Goal:** Stop distributing AWS/Pinecone keys to clients.
*   **Action:** Build a central API server (e.g., Python FastAPI).
*   **Flow:**
    *   Client sends data to `api.project_name.ai`.
    *   Server validates request.
    *   Server talks to S3/Pinecone.

## 2. 👤 User Authentication
*   **Goal:** Manage multi-tenant access.
*   **Action:** Integrate Auth0, Supabase, or AWS Cognito.
*   **Impact:** CLI will need a `login` command: `project_name login`.

## 3. 💳 Billing & Credits
*   **Goal:** Monetize the service.
*   **Database:** A PostgreSQL table tracking `user_id`, `credits_balance`.
*   **Logic:**
    *   `Upload` event -> +1 Credit.
    *   `Download` event -> -2 Credits.
*   **Integration:** Stripe API for "Buy Credits".

## 4. 🌍 The Shared Marketplace
*   **Goal:** Allow users to search the global pool.
*   **Action:** Add a `visibility` field to Pinecone vectors (`private` vs `public`).
*   **Search Logic:**
    *   Private Search: Filter `owner_id == current_user`.
    *   Global Search: Filter `visibility == public`.

## 5. 🔒 Encryption
*   **Goal:** Enterprise-grade privacy.
*   **Action:** Encrypt image blobs in S3 so even project_name admins cannot view private data without the user's key.
