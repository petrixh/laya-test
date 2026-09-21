"""Hand-authored graded probes.

These are not accuracy tests. Each ladder is ordered by intended intensity, and
we check the model's scores move monotonically with it. That is what catches the
"soft signals are underweighted" failure a single pass/fail assertion misses.
"""
from __future__ import annotations

from .runner import Service, kendall_tau

# rank 0 = strongest signal. Ordered deliberately, strongest -> weakest.
CHURN_LADDER = [
    ("explicit_cancel", "We are cancelling our subscription effective immediately. Please confirm."),
    ("notice_given", "Please close our account at the end of the current billing period."),
    ("ultimatum", "If this is not fixed by Friday we will terminate the contract."),
    ("vendor_shopping", "Given these repeated problems we have started evaluating other vendors."),
    ("renewal_doubt", "We are no longer sure we will renew when the contract comes up."),
    ("frustrated", "This is the third outage this month and we are losing patience."),
    ("mild_complaint", "The dashboard has felt a little slow this week."),
    ("neutral", "Could you confirm the date of our next invoice?"),
    ("happy", "The new release is excellent. We just renewed for another two years."),
]

URGENCY_LADDER = [
    ("catastrophic", "Production is completely down, all customer payments are failing right now."),
    ("severe", "A major feature is broken for most of our users and we need it fixed today."),
    ("moderate", "A report is rendering incorrectly; we would like it sorted this week."),
    ("minor", "There is a small typo on the settings page whenever you get a chance."),
    ("none", "Just curious how often the changelog page is updated."),
]

CHURN_Q = {"churn_risk": {"type": "noul", "instructions": "Does the user express risk of cancelling or leaving?"}}
URGENCY_Q = {"urgency": {"type": "score", "instructions": "How urgent is this request?",
                         "criteria": ["not urgent", "soon", "critical deadline"]}}


def run_ladder(svc: Service, ladder, questions, key, field) -> dict:
    rows = []
    for rank, (name, text) in enumerate(ladder):
        res = svc.predict(text, questions)
        rows.append({"rank": rank, "case": name, "value": res["answers"][key][field],
                     "confidence": res["answers"][key].get("confidence")})

    # ladder is strongest-first, so intended value ordering is descending in rank
    tau = kendall_tau([-r["rank"] for r in rows], [r["value"] for r in rows])
    values = [r["value"] for r in rows]
    inversions = [
        (rows[i]["case"], rows[j]["case"], round(values[i], 4), round(values[j], 4))
        for i in range(len(rows)) for j in range(i + 1, len(rows))
        if values[i] < values[j]
    ]
    return {
        "key": key,
        "field": field,
        "kendall_tau": round(tau, 4),
        "spread": round(max(values) - min(values), 4),
        "rows": rows,
        "inversions": inversions,
    }


def run_all(svc: Service) -> dict:
    return {
        "churn": run_ladder(svc, CHURN_LADDER, CHURN_Q, "churn_risk", "noul"),
        "urgency": run_ladder(svc, URGENCY_LADDER, URGENCY_Q, "urgency", "score"),
    }
