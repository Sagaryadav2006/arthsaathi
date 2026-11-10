# 🪙 ArthSaathi (अर्थसाथी)

**Enterprise Digital Financial Literacy & Cyber Safety AI Agent**

ArthSaathi is a robust, production-ready AI agent designed to bridge the digital financial literacy gap in India. Built with a highly resilient architecture, it offers hallucination-free financial education, real-time threat analysis, official government scheme discovery, and strict regulatory grounding.

## 🚀 Core Features & Architecture

- 💬 **Cascading RAG Financial Assistant**
  * **Tier 1 (Local):** Queries a local FAISS vector database containing official Master Circulars (e.g., RBI Zero Liability).
  * **Tier 2 (Live Web Fallback):** Automatically intercepts out-of-bounds general finance queries and performs a live, domain-restricted web search (`rbi.org.in`, `sebi.gov.in`, `ncfe.org.in`), expanding the context dynamically while maintaining strict official grounding.

- 🛡️ **Phishing & SMS Threat Analyzer**
  * Scans suspicious messages, APK links, and emails for psychological manipulation and credential harvesting. Features flawless Streamlit state-management for UI rendering and resetting.

- 📜 **Official Guidelines Decoder**
  * Ingests raw regulatory text directly from central authorities via the **Serper.dev Google Search API**.
  * Implements strict **Domain Fencing** (e.g., `site:rbi.org.in OR site:sebi.gov.in`) to categorically eliminate global SEO hallucinations and generic tech-support answers.
  * Utilizes a custom **Aggressive Regex Sanitization Pipeline** to strip SERP metadata (relative timestamps, dates, footer boilerplate, obfuscated emails).
  * Synthesizes complex legal jargon into a 5th-grade pedagogical format: **Core Rule, Explanation, and Real-World Example**.

- 🏛️ **Strict Government Scheme Finder**
  * Matches users with verified central and state welfare/subsidy programs via keyword-optimized programmatic scraping of `myscheme.gov.in` and `india.gov.in`.
  * Enforces a **Strict Ownership Guardrail** at the LLM level, actively filtering out and rejecting private banks, NBFCs, and commercial loans to guarantee safe, ad-free recommendations.

- 🧮 **Precision Financial Calculators**
  * High-accuracy EMI and SIP calculators built with robust Streamlit input widgets to eliminate UI state conflicts.

## ⚙️ Enterprise Data Pipelines & Cloud Resilience

ArthSaathi was engineered to survive production-grade cloud environments and third-party rate limits:

- **Hybrid SERP Engine:** Overcomes shared-IP rate limiting on Streamlit Community Cloud. Utilizes the robust **Serper.dev API** for high-volume structured data retrieval (Tabs 3 & 6), while maintaining an **Anti-Gravity Scraper** (DDGS + `urllib`/`BeautifulSoup` User-Agent spoofing) as a failsafe for organic conversational RAG (Tab 1).
- **Offline Embedding Model Bundling:** Completely bypasses Hugging Face Hub download rate limits (`LocalEntryNotFoundError`) by bundling the `sentence-transformers/all-MiniLM-L6-v2` model directly into the containerized local environment for zero-latency FAISS mounting.
- **Centralized LLM Factory:** Uses `get_primary_google_llm` equipped with auto-retries, strict timeouts, and a seamless offline IBM Granite (`watsonx`) LLM fallback in case of primary API outages.
- **Golden Hour Panic Detection:** Immediately detects emergency keywords ("stolen", "fraud") and triggers a high-visibility UI intercept displaying the National Cyber Helpline (1930) *before* processing standard advice.
- **Zero-Leak PII Redaction:** A pre-processing regex firewall that redacts phone numbers, UPI PINs, and sensitive financial identifiers before they ever reach the LLM.

## 💻 Tech Stack

- **Frontend:** Streamlit
- **AI/LLM:** Google Gemini 1.5 Flash, IBM Watsonx Granite 
- **Orchestration:** LangChain, LangChain-Core
- **Vector Database:** FAISS (CPU)
- **Embeddings:** Local HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)
- **Web Extraction:** Serper.dev Google Search API, DuckDuckGo-Search (`ddgs`), BeautifulSoup4, Requests
- **Data Parsing:** Ast Literal Eval, Custom Regex Sanitizers

## 🔐 Disclaimer

ArthSaathi is an educational tool designed to promote digital financial literacy and cyber safety. It strictly extracts and summarizes information from official Indian government and regulatory portals. It does not provide personalized investment advice, stock tips, or direct loan approvals. Always consult official bank portals or the National Cyber Crime Portal (1930) for emergency assistance.
