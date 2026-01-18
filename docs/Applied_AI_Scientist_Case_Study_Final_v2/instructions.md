
📝 **Important:** Your system architecture should represent how you'd build this in production. However, your prototype can be built locally without cloud infrastructure. We're more interested in your design thinking and practical trade-offs than complete integration.
# 🧱 Architecture Instructions

Please include the following in your system architecture document (PDF or slides):

## 1. System Diagram
- Visual diagram of how components interact (users, frontend, backend, ML/AI, database, cloud)
- Can be created in draw.io, Excalidraw, Figma, etc.

## 2. Description of Azure Setup
- What services you would use (App Services, Blob Storage, Cosmos DB, Azure AI Search, etc.)
- How each service maps to your architecture

## 3. API Contracts
- Key request/response schemas for user input and recommendations
- Sample payloads or OpenAPI-style docs (if preferred)

## 4. AI & Agent Integration
- If using LLMs, describe which agents exist and how they interact
- Define how multi-agent logic improves recommendations

## 5. Edge Case Handling
- Timeout or fallback behavior
- Rate limiting / abuse prevention
- Security considerations (e.g., input validation, data access control, etc.)
