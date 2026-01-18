# 🏗️ Senior Case Study: AI-Powered Property Recommendation System

Welcome! This case study is designed to evaluate your ability to design and prototype a scalable, intelligent full stack application for Agent Mira.

---

## 🎯 Objective

Build a working MVP of an AI-powered property recommendation system where users input preferences and receive 2–3 recommended properties with reasoning.


> ⚠️ **Note:** While we expect your architecture document to reflect how you'd build this as a production-grade, scalable system (including Azure components and AI frameworks), your actual implementation can be lightweight and local. You may simulate cloud services or use simple Python scripts — the goal is to assess how you think, not whether you have Azure credits.

---

## 🧩 Core Tasks

### 1. 🔧 Prototype App

- **Frontend**:
  - Form to capture buyer preferences: budget, city, min beds, max commute time, etc.
  - Display the top 2–3 recommended properties and reasoning behind each

- **Backend**:
  - Accepts input from frontend
  - Fetches or simulates a list of properties
  - Applies custom scoring or uses a provided ML model (`complex_price_model_v2.pkl`)
  - Returns ranked results with explanations

### 2. 🧱 Architecture Document

- End-to-end system design with component breakdown:
  - Frontend, backend, AI layer, data store, cloud infra
- APIs and user interaction flows
- Azure services: App Services, Cosmos DB, Azure AI Search, etc.
- Where and how LLM/Agentic AI fits in (LangChain, CrewAI, etc.)
- Considerations for scalability, error handling, and **security best practices**

---

## 🌟 Bonus (Optional)

- Multi-agent collaboration logic (e.g., budget agent + location agent)
- Real-time refinement loop based on user feedback
- Additional filtering or explainability logic

---

## 🧠 What We’re Looking For

| Capability               | Signals We're Watching For                     |
|--------------------------|------------------------------------------------|
| Technical depth          | Clean, modular code and backend structure      |
| Design thinking          | Scalable, secure, and realistic architecture   |
| AI maturity              | Practical use of LLM agents or search tools    |
| Product mindset          | Focus on usefulness and edge-case thinking     |
| Ownership                | Ability to anticipate and handle real-world issues |

---

## 📦 Files Provided

- `complex_price_model_v2.pkl`: Optional price scoring model for ranking
- `docs/instructions.md`: This case brief
- `backend/main.py`: Starter FastAPI script to use or extend

---

## ✅ Submission Instructions

- GitHub repo (or zipped folder) with working prototype
- `README.md` with:
  - Setup instructions
  - Overview of architecture decisions
- PDF or slides with system architecture and flow diagrams

We’re excited to see how you approach the challenge!

