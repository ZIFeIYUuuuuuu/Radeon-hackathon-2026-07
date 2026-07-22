"""Generate a synthetic, local-only SFT corpus for ClaimCourt behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM = (
    "You are ClaimCourt, a private local evidence and file-location agent. "
    "Return valid JSON only. Never invent citations. Never reveal a secret value; "
    "return a redacted preview and a fingerprint instead."
)


def record(user: str, assistant: dict[str, object]) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(assistant, separators=(",", ":"))},
        ]
    }


def build_records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    vendors = ["Atlas", "Brightline", "Cobalt", "Delta", "Evergreen", "Fjord"]
    commitments = ["99.9% uptime", "four-hour response", "a fixed delivery date"]
    for index in range(72):
        vendor = vendors[index % len(vendors)]
        commitment = commitments[index % len(commitments)]
        mode = index % 3
        if mode == 0:
            evidence = f"[ORDER-{index:02d}] Signed order form: {vendor} shall provide {commitment}."
            verdict = "supported"
            reason = "A signed order form directly states the commitment."
            citations = [f"ORDER-{index:02d}"]
            missing = []
        elif mode == 1:
            evidence = (
                f"[MSA-{index:02d}] The agreement creates no service level commitment unless a separately signed addendum exists.\n"
                f"[EMAIL-{index:02d}] Sales wrote: enterprise-grade reliability."
            )
            verdict = "contradicted"
            reason = "The binding agreement rejects an SLA while the email is non-binding marketing."
            citations = [f"MSA-{index:02d}", f"EMAIL-{index:02d}"]
            missing = ["A signed addendum or order form"]
        else:
            evidence = (
                f"[NOTES-{index:02d}] We target {commitment} after Q4, subject to capacity.\n"
                f"[EMAIL-{index:02d}] Sales described the service as reliable."
            )
            verdict = "insufficient_evidence"
            reason = "A conditional target and marketing language do not establish a binding commitment."
            citations = [f"NOTES-{index:02d}", f"EMAIL-{index:02d}"]
            missing = ["A signed contractual commitment"]
        rows.append(
            record(
                f"Decide whether {vendor} committed to {commitment}. Evidence packet:\n{evidence}",
                {
                    "route": "evidence_court",
                    "claim": f"{vendor} committed to {commitment}",
                    "verdict": verdict,
                    "confidence": 0.86,
                    "evidence_citations": citations,
                    "reasoning": reason,
                    "missing_evidence": missing,
                    "recommended_next_action": "Request the controlling signed record." if missing else "Use the signed order form in the decision brief.",
                },
            )
        )

    route_examples = [
        ("Find the slide deck about our pricing proposal.", "file_locator", ["pptx", "pdf"], False),
        ("Where did I leave a private key?", "sensitive_record_scan", ["all"], False),
        ("Did the customer approve the delay?", "evidence_court", ["email", "docx", "pdf"], False),
        ("Locate the meeting notes that mention Project Northstar.", "file_locator", ["md", "docx", "txt"], False),
        ("Search for API credentials in this workspace.", "sensitive_record_scan", ["all"], False),
        ("Create a brief explaining who approved the change request.", "evidence_court", ["email", "pdf", "docx"], True),
    ]
    for index in range(54):
        query, route, file_types, export_requested = route_examples[index % len(route_examples)]
        rows.append(
            record(
                query,
                {
                    "route": route,
                    "search_scope": "selected_local_workspace",
                    "file_types": file_types,
                    "requires_export_approval": export_requested,
                    "privacy_rule": "never transmit source contents outside the local runtime",
                },
            )
        )

    for index in range(36):
        token = f"ghp_EXAMPLE{index:02d}ABCDEFGHIJKLMN1234567890"
        rows.append(
            record(
                f"A local scan found `{token}` in secrets/notes-{index:02d}.txt on line 4. Report it safely.",
                {
                    "route": "sensitive_record_scan",
                    "findings": [
                        {
                            "source": f"secrets/notes-{index:02d}.txt",
                            "locator": "line 4",
                            "kind": "github token",
                            "fingerprint": f"sha256:example{index:02d}fingerprint",
                            "redacted_preview": "ghp_...7890",
                        }
                    ],
                    "raw_secret_disclosed": False,
                    "recommended_next_action": "Revoke or rotate the credential and remove it from the workspace.",
                },
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("training/data/claimcourt_sft.jsonl"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = build_records()
    args.output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} synthetic examples to {args.output}")


if __name__ == "__main__":
    main()
