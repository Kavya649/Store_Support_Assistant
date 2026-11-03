import re
from datetime import timedelta, timezone

import streamlit as st
from dotenv import load_dotenv

from backend import FIXED_REFERENCE, load_records, verify
from llm import DEFAULT_MODEL, LLMError, classify_query, draft_reply

load_dotenv()
st.set_page_config(page_title="Store Support Assistant", page_icon="🛍️", layout="wide")
st.title("🛍️ Store Support Assistant")
st.caption("LLM-assisted query interpretation · deterministic record checks · grounded customer replies")

_ref_utc = FIXED_REFERENCE.strftime("%d %b %Y, %H:%M UTC")
_ref_ist = FIXED_REFERENCE.astimezone(timezone(timedelta(hours=5, minutes=30))).strftime("%H:%M IST")

with st.sidebar:
    st.subheader("Configuration")
    st.write("Place the three CSV files in the `data/` folder.")
    st.write("Uses a local Llama model through Ollama (no API key needed).")
    st.write(f"Fixed delivery reference: **{_ref_utc}** ({_ref_ist}).")
    model = st.text_input("Model", value=DEFAULT_MODEL)

query = st.text_area("Customer query", placeholder="e.g. Was I charged twice for ORD-8101?", height=130)
submit = st.button("Analyze query", type="primary", disabled=not query.strip())

if submit:
    try:
        # LLM interprets the query; Python independently validates the ID and checks records.
        classification = classify_query(query, model=model)
        if classification["issue"] == "OUT_OF_SCOPE":
            result = {"status": "OUT_OF_SCOPE", "issue": None, "order_id": None,
                      "explanation": "The query is outside the supported payment, refund, and delivery use cases.",
                      "evidence": []}
        else:
            # The regex-found ID is the source of truth; the model's ID is only a cross-check.
            found_ids = {x.upper() for x in re.findall(r"\bORD-\d{4}\b", query, re.IGNORECASE)}
            model_id = classification.get("order_id")
            if len(found_ids) != 1:
                result = {"status": "NEEDS_INFORMATION", "issue": classification["issue"], "order_id": None,
                          "explanation": "Please provide exactly one order ID in the format ORD-1234.",
                          "evidence": []}
            else:
                order_id = next(iter(found_ids))
                if model_id and model_id != order_id:
                    result = {"status": "NEEDS_INFORMATION", "issue": classification["issue"], "order_id": order_id,
                              "explanation": "The order ID could not be confirmed. Please restate the order ID you are asking about.",
                              "evidence": []}
                else:
                    records = load_records("data")
                    v = verify(classification["issue"], order_id, records)
                    result = {"status": v.status, "issue": v.issue, "order_id": v.order_id,
                              "explanation": v.explanation, "evidence": v.evidence}

        if result["status"] == "ERROR":
            st.error(result["explanation"])
        else:
            reply = draft_reply(query, result, model=model)
            left, right = st.columns([1.15, 1])
            with left:
                st.subheader("Customer-facing reply")
                st.success(reply)
            with right:
                st.subheader("Verification result")
                st.metric("Status", result["status"])
                st.write("**Issue:**", result.get("issue") or "—")
                st.write("**Order:**", result.get("order_id") or "—")
                st.write(result["explanation"])
                st.subheader("Supporting records")
                if result["evidence"]:
                    st.dataframe(result["evidence"], hide_index=True)
                else:
                    st.info("No evidence rows are attached to this status.")
    except FileNotFoundError as e:
        st.error(f"Data setup error: {e}")
    except ValueError as e:
        st.error(f"Data format error: {e}")
    except LLMError as e:
        st.error(f"LLM error: {e}")
    except Exception as e:
        st.error(f"Unexpected application error: {e}")
        st.caption("Check the terminal logs for debugging details. No customer reply was generated from an unverified result.")

with st.expander("Supported issue types"):
    st.markdown("""
    - **DOUBLE_CHARGE** — two SUCCESS payments with equal amount and different nonblank provider references.
    - **PAYMENT_FAILURES** — three FAILED attempts spanning at most ten minutes, earliest to latest.
    - **REFUND_STATUS** — report the current refund record without inferring bank receipt or arrival date.
    - **DELIVERY_STATUS** — compare IN_TRANSIT deadline against the fixed UTC reference; DELIVERED uses the recorded time.
    """)
