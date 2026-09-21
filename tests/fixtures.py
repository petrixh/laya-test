"""Shared question sets and labelled cases.

Cases are deliberately unambiguous: the point is to prove the model is wired up
and behaving sanely, not to measure accuracy on hard examples.
"""

TRIAGE_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
}

# (case id, state, expected department)
ROUTING_CASES = [
    (
        "billing_refund",
        {
            "from": "user@acme.com",
            "subject": "Duplicate charge on invoice #4411",
            "body": "Hi, we were billed twice for March. Please refund the extra charge.",
        },
        "billing",
    ),
    (
        "technical_outage",
        {
            "from": "dev@acme.com",
            "subject": "500 errors from the API since 09:00",
            "body": "Every request to /v1/orders returns HTTP 500. Our production integration is broken.",
        },
        "technical",
    ),
    (
        "sales_pricing",
        {
            "from": "cto@newco.io",
            "subject": "Enterprise pricing for 400 seats",
            "body": "We are evaluating vendors and would like a quote and a contract draft for 400 seats.",
        },
        "sales",
    ),
]

URGENT_STATE = {
    "subject": "PRODUCTION DOWN - payments failing",
    "body": "All customer payments have been failing for 40 minutes. We are losing revenue right now. Need this fixed immediately.",
}
CASUAL_STATE = {
    "subject": "Question about the docs",
    "body": "No rush at all, just curious whether the changelog page is updated monthly or quarterly.",
}

CANCEL_STATE = {
    "subject": "Cancelling our account",
    "body": "We have decided to cancel our subscription at the end of the month. Please confirm the cancellation.",
}
HAPPY_STATE = {
    "subject": "Thanks for the great support",
    "body": "Just wanted to say the new dashboard is excellent. We are renewing for another two years.",
}
