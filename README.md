# Customer Support Assistant (LLM + SQL)

This project implements an intelligent **customer support chatbot** capable of answering user questions about their past or current e-commerce orders.  
It combines a **Large Language Model (LLM)** with a **SQL database** to provide real-time, contextual responses about order status, shipping dates, or payments.

---

## Project Overview

The main objective is to **automate customer service interactions** by providing quick, accurate, and personalized answers — while reducing the workload of human agents.

### The assistant can:
- Understand and classify user questions in natural language (NLP intent classification)
- Query a relational **SQLite database** containing user and order information
- Generate contextual and natural replies based on retrieved data
- Escalate to human support when appropriate (e.g., modification or cancellation requests)

---

## Functional Description

When a user sends a message, the system goes through three key LLM-based reasoning steps:

1. **Intent Classification**  
   Determines the user’s intent among:
   - `ORDER_INFO` (requesting status)
   - `ORDER_HELP` (asking for human assistance)
   - `OUT_OF_SCOPE` (irrelevant queries)

2. **Parameter Extraction**  
   Identifies the related order ID (if mentioned or implied, e.g. “my last order”)  
   and checks whether clarification is needed.

3. **SQL Query + Final Answer Generation**  
   Retrieves the order’s information from the database and crafts a final response that is:
   - **Accurate** (based on real order data)
   - **Friendly** (tone consistent with customer support)
   - **Safe** (no data leaks between users)

---

## Technical Details

### Main Technologies
- **Python 3.10+**
- **Transformers** (Hugging Face)
- **Streamlit** (UI)
- **SQLite** (data storage)
- **Logging** (structured runtime logs)
- **LangChain-style prompting** (custom lightweight implementation)

### Model
- Default model: `microsoft/Phi-3-mini-4k-instruct`
- Target production model: `mistralai/Mistral-7B-Instruct-v0.3`

The model is loaded via the `transformers` library using the **Apple MPS** backend or CPU fallback.  
For local development, short inference chains are used (`SMALL_KW`, `MED_KW`, `LONG_KW`) to improve latency.

---

## Usage

```bash
# Activate virtuaal environnement
source venv/bin/activate
# install dependencies
pip install -r requirements.txt
# Download example database
curl -O https://blent-learning-user-ressources.s3.eu-west-3.amazonaws.com/projects/e23c6b/data/orders.db
# Run application
streamlit run src/app.py
```

## Constraints & Security

- Each user is authenticated (email, name, and user ID are known).  
- The chatbot never exposes or accesses another user's orders.  
- Input sanitization is performed to prevent prompt injection attacks.  
- Semantic routing ensures that only service-related queries are processed by the LLM.  
- If the detected intent is `ORDER_HELP`, the assistant politely transfers the conversation to a human agent.  
- Access control checks are applied before executing any SQL query to prevent unauthorized data exposure.

---

## Mac (Apple Silicon) Compatibility Notes

Because this project is developed on macOS (Apple Silicon, M1/M2/M3), several technical constraints apply:

- CUDA is not available on Mac, and therefore the quantization method used by `bitsandbytes` (`load_in_8bit` / `load_in_4bit`) cannot be used.  
- Loading a large model such as Mistral-7B in full precision exceeds the memory available on most local systems (approximately 13–14 GiB).
- Consequently, smaller models (around 3–4B parameters, e.g. `Phi-3-mini`) are used for **local development and testing**.  
- For production deployment, Mistral-7B can be used on a CUDA-enabled GPU instance

### Summary

| Platform | Quantization Support | CUDA | MPS (Metal) | Recommended Model |
|-----------|----------------------|------|--------------|-------------------|
| macOS (local dev) | Not supported via bitsandbytes | No | Yes | Phi-3-mini (3B) |
| Cloud GPU | Fully supported | Yes | No | Mistral-7B |

---

## Future Improvements
- Deployment as a microservice using **FastAPI** and **Docker**
- Connection to a live e-commerce database (MySQL or PostgreSQL)
- User authentication and session management
- Caching of recent responses for faster performance
- Integration of a lightweight **RAG component** for FAQ-style questions
- Automated evaluation and performance tracking on real support data

---

## Author

Rémi Nollet | [LinkedIn](www.linkedin.com/in/remi-nollet)
AI Engineer specialized in Computer Vision and LLM Engineering
