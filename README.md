# Store Support Assistant — Groq + Llama

A Streamlit support tool that uses **Groq-hosted Llama 3.3 70B** to interpret customer queries and draft concise replies. Python independently verifies the fictional CSV records; the LLM is never treated as evidence.

## 1. Requirements
- Python 3.10+
- A Groq API key from https://console.groq.com/keys
- The three supplied CSV files

## 2. Project structure

```text
store_support_assistant/
├── app.py
├── backend.py
├── llm.py
├── requirements.txt
├── .env.example
├── README.md
└── data/
    ├── payments.csv
    ├── refunds.csv
    └── deliveries.csv
```

Put the supplied CSV files in `data/`. Do not modify their contents.

## 3. Create and activate a virtual environment (Windows)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If using Command Prompt instead:

```cmd
venv\Scripts\activate.bat
```

## 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Configure Groq

1. Create an API key at https://console.groq.com/keys.
2. Copy `.env.example` to a new file named `.env` in the project root.
3. Replace `your_groq_api_key_here` with your key. Never commit or share `.env`.

```env
GROQ_API_KEY=your_actual_key
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TIMEOUT=45
```

The model ID is configurable in `.env` or the Streamlit sidebar. Check https://console.groq.com/docs/models for currently available model IDs.

## 6. Run

```bash
streamlit run app.py
```

## 7. Processing flow

1. Employee enters one customer query.
2. Groq Llama classifies the issue and extracts an order ID.
3. Python checks the exact ID requirement and verifies the relevant CSV records.
4. Python determines the authoritative result status and evidence.
5. Groq Llama drafts a short customer-facing reply using only the verified result.
6. UI displays the reply, status, explanation, and evidence separately.

## Supported cases
- `DOUBLE_CHARGE`: two SUCCESS rows with equal amount and distinct nonblank provider references.
- `PAYMENT_FAILURES`: three FAILED attempts within ten minutes, inclusive.
- `REFUND_STATUS`: current refund record and recorded fields.
- `DELIVERY_STATUS`: uses fixed reference time 2026-11-02 12:00 UTC.

## Safety and correctness
- Model output is interpretation, not evidence.
- Python checks all payment/refund/delivery facts.
- Exactly one distinct order ID must appear in the query and match the model interpretation.
- Missing records and ambiguous IDs return `NEEDS_INFORMATION` with no evidence.
- LLM failures return a clear error; no reply is drafted from unverified facts.
- Do not put real customer payment data or secrets in an unapproved third-party service. This assignment uses fictional records.
