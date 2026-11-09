import pytest
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import (
    redact_pii,
    check_regulatory_guardrails,
    check_emergency_panic,
    build_fallback_rag_chain,
    SCAM_SCENARIOS,
    generate_new_scenarios,
    analyze_threat_content,
    heuristic_threat_analyzer,
    simplify_regulatory_jargon,
    fetch_live_schemes,
    ResilientGraniteLLM
)

class TestPIIRedaction:
    def test_mobile_number_redaction(self):
        text = "Contact me at 9876543210 for details."
        sanitized, redactions = redact_pii(text)
        assert "[REDACTED_PHONE_NUMBER]" in sanitized
        assert "9876543210" not in sanitized
        assert len(redactions) > 0

    def test_upi_pin_redaction(self):
        text = "My UPI PIN is 123456 please check."
        sanitized, redactions = redact_pii(text)
        assert "[REDACTED_SECURITY_CODE]" in sanitized or "[REDACTED_UPI_PIN]" in sanitized
        assert "123456" not in sanitized

    def test_aadhaar_redaction(self):
        text = "My Aadhaar number is 5432 1234 8765."
        sanitized, redactions = redact_pii(text)
        assert "[REDACTED_AADHAAR]" in sanitized
        assert "5432 1234 8765" not in sanitized

    def test_card_number_redaction(self):
        text = "Card number is 4532-1234-5678-9012"
        sanitized, redactions = redact_pii(text)
        assert "[REDACTED_CARD_NO]" in sanitized or "[REDACTED_AADHAAR]" in sanitized
        assert "4532-1234-5678-9012" not in sanitized

    def test_email_redaction(self):
        text = "Send statement to citizen@example.com."
        sanitized, redactions = redact_pii(text)
        assert "[REDACTED_EMAIL]" in sanitized
        assert "citizen@example.com" not in sanitized


class TestRegulatoryGuardrails:
    def test_investment_advice_refusal(self):
        query = "Which stock should I buy for quick 50% gain?"
        refusal = check_regulatory_guardrails(query)
        assert refusal is not None
        assert "SEBI-registered Investment Adviser" in refusal
        assert "prohibited by regulators" in refusal

    def test_loan_approval_refusal(self):
        query = "Can you approve my loan of 50000 rupees?"
        refusal = check_regulatory_guardrails(query)
        assert refusal is not None
        assert "cannot evaluate, approve, or sanction loans" in refusal
        assert "RBI-regulated bank" in refusal

    def test_benign_query_allowed(self):
        query = "What is the zero liability period under RBI bank fraud circular?"
        refusal = check_regulatory_guardrails(query)
        assert refusal is None


class TestScamScenariosAndGenerator:
    REQUIRED_KEYS = {
        "id", "title", "category", "sender", "message",
        "options", "correct_index", "psychological_trigger",
        "regulatory_rule", "immediate_action"
    }

    def test_static_scenarios_schema(self):
        assert len(SCAM_SCENARIOS) >= 5
        for s in SCAM_SCENARIOS:
            assert set(s.keys()) == self.REQUIRED_KEYS
            assert isinstance(s["id"], int)
            assert isinstance(s["options"], list)
            assert len(s["options"]) == 3
            assert s["correct_index"] in (0, 1, 2)

    def test_generator_schema_and_id_sequencing(self):
        current_count = 5
        new_cases = generate_new_scenarios(current_count)
        assert len(new_cases) == 5
        assert [c["id"] for c in new_cases] == [6, 7, 8, 9, 10]
        for c in new_cases:
            assert set(c.keys()) == self.REQUIRED_KEYS
            assert len(c["options"]) == 3
            assert c["correct_index"] in (0, 1, 2)


class TestPhishingAndThreatAnalyzer:
    def test_heuristic_apk_threat(self):
        msg = "Electricity bill overdue. Disconnection tonight! Download http://statepower.apk"
        result = heuristic_threat_analyzer(msg)
        assert result["risk_score"] >= 75
        assert result["verdict"] == "CRITICAL RISK"
        assert len(result["red_flags"]) >= 2
        assert len(result["guidance"]) >= 2

    def test_analyze_threat_content_fallback(self):
        msg = "Your account is credited with Rs 500 from salary."
        result = analyze_threat_content(msg)
        assert "risk_score" in result
        assert "verdict" in result
        assert "red_flags" in result
        assert "guidance" in result


class TestFinancialCalculations:
    def test_emi_calculation_formula(self):
        principal = 500000.0
        annual_rate = 10.5
        tenure_months = 60
        r = (annual_rate / 100.0) / 12.0
        emi = (principal * r * ((1 + r) ** tenure_months)) / (((1 + r) ** tenure_months) - 1)
        total_payable = emi * tenure_months
        total_interest = total_payable - principal

        assert round(emi, 2) == 10746.95
        assert round(total_payable, 2) == 644817.01
        assert round(total_interest, 2) == 144817.01

    def test_sip_calculation_formula(self):
        monthly = 5000.0
        annual_return = 12.0
        tenure_months = 120
        i = (annual_return / 100.0) / 12.0
        future_value = monthly * (((1 + i) ** tenure_months - 1) / i) * (1 + i)
        total_invested = monthly * tenure_months
        wealth_gain = future_value - total_invested

        assert total_invested == 600000.0
        assert round(future_value, 2) == 1161695.38
        assert round(wealth_gain, 2) == 561695.38


class TestPhase6Features:
    def test_simplify_regulatory_jargon_fallback(self):
        sample_text = (
            "Reserve Bank of India Circular on Limiting Liability: Zero Customer Liability applies when "
            "unauthorised electronic transactions occur provided the customer notifies the bank within three (3) working days."
        )
        simple_res = simplify_regulatory_jargon(sample_text)
        assert isinstance(simple_res, str)
        assert len(simple_res) > 30
        assert "3 days" in simple_res or "bank" in simple_res.lower()

    def test_fetch_live_schemes_schema_and_fallback(self):
        schemes = fetch_live_schemes(
            age=32,
            income="Below ₹2,50,000",
            occupation="Farmer / Agricultural Worker",
            specific_need="Pension"
        )
        assert isinstance(schemes, list)
        assert len(schemes) >= 1
        for sch in schemes:
            assert "Scheme_Name" in sch
            assert "Eligibility" in sch
            assert "Core_Benefit" in sch
            assert "Purpose_And_Working" in sch
            assert "Official_Portal_Link" in sch
            assert str(sch["Official_Portal_Link"]).startswith("http")

    def test_fetch_live_schemes_insurance(self):
        schemes = fetch_live_schemes(
            age=25,
            income="₹2,50,000 - ₹5,00,000",
            occupation="Salaried Employee (Private / Govt)",
            specific_need="Life and Accident Insurance"
        )
        assert isinstance(schemes, list)
        assert any("Insurance" in s["Scheme_Name"] or "Bima" in s["Scheme_Name"] or "Jeevan" in s["Scheme_Name"] or s["Scheme_Name"] == "Live Search Unavailable" for s in schemes)

    def test_aggressive_ast_literal_eval_extraction(self):
        import ast
        # Test case: raw stringified list with dictionary metadata signature
        raw_content = "[{'type': 'text', 'text': 'Zero customer liability applies within 3 working days.'}]"
        assert raw_content.strip().startswith("[") and raw_content.strip().endswith("]")
        parsed_list = ast.literal_eval(raw_content.strip())
        assert isinstance(parsed_list, list)
        extracted = "".join(chunk.get("text", "") for chunk in parsed_list if isinstance(chunk, dict) and "text" in chunk)
        assert extracted == "Zero customer liability applies within 3 working days."

    def test_resilient_granite_clean_offline_parser(self):
        granite = ResilientGraniteLLM()
        sample_prompt = (
            "OFFICIAL GUIDELINES CONTEXT:\n* Zero customer liability applies within 3 working days.\n\n"
            "USER QUESTION:\nHow do I report unauthorized UPI debit?\n\n"
            "Grounded & Structured Response:"
        )
        result = granite._generate([sample_prompt])
        assert len(result.generations) == 1
        output_text = result.generations[0][0].text
        assert "⚠️ *Operating in Secure Offline Mode due to API unavailability.*" in output_text
        assert "**Official Guidance for your query:**" in output_text
        assert "Zero Liability Protection" in output_text
        assert "Essential Safety Actions" in output_text

    def test_check_emergency_panic_precision_triggers(self):
        # Benign questions containing 'fraud' or 'unauthorized' must NOT trigger panic alert
        assert check_emergency_panic("What is the zero liability period under RBI bank fraud circular?") is False
        assert check_emergency_panic("Tell me about unauthorized transactions rules.") is False

        # Active loss phrases MUST trigger panic alert
        assert check_emergency_panic("I was scammed on WhatsApp yesterday") is True
        assert check_emergency_panic("My bank account drained completely") is True
        assert check_emergency_panic("Money lost in cyber theft") is True
        assert check_emergency_panic("Transferred by mistake to stranger") is True
        assert check_emergency_panic("Someone withdrew money from ATM") is True

    def test_build_fallback_rag_chain_transparent_objects(self):
        primary_llm, fallback_llm, prompt, status = build_fallback_rag_chain()
        assert primary_llm is not None
        assert fallback_llm is not None
        assert prompt is not None
        assert isinstance(status, str)
