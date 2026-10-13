"""Save ForwardTrue AI reports, claims, and evidence to Supabase."""

import os
import math
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


def _get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        raise RuntimeError(
            "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env"
        )

    return create_client(url, key)


def _safe_score(value):
    try:
        score = float(value)
        return score if math.isfinite(score) else None
    except (TypeError, ValueError):
        return None


def save_report_to_supabase(report, extracted_text, metadata):
    """Insert one report and its child claims/evidence. Return report UUID."""

    client = _get_supabase_client()
    metadata = metadata or {}
    provenance = report.get("provenance") or {}

    report_row = {
        "input_type": metadata.get("input_type") or "unknown",
        "input_source": (
            metadata.get("source")
            or provenance.get("original_source")
        ),
        "input_title": (
            metadata.get("title")
            or provenance.get("original_title")
        ),
        "model": report.get("model"),
        "status": report.get("status"),
        "extracted_text": extracted_text,
        "provenance": provenance,
        "report_json": report,
    }

    response = (
        client.table("fact_check_reports")
        .insert(report_row)
        .execute()
    )

    if not response.data:
        raise RuntimeError("Supabase did not return the inserted report.")

    report_id = response.data[0]["id"]

    try:
        for index, item in enumerate(report.get("claims", []), start=1):
            claim_row = {
                "report_id": report_id,
                "claim_index": index,
                "claim_text": item.get("claim", ""),
                "context": item.get("context") or "",
                "verdict": item.get("verdict", "Insufficient Evidence"),
                "confidence": item.get("confidence", "low"),
                "explanation": item.get("explanation") or "",
                "limitations": item.get("limitations") or [],
                "citations": item.get("citations") or [],
            }

            claim_response = (
                client.table("fact_check_claims")
                .insert(claim_row)
                .execute()
            )

            if not claim_response.data:
                raise RuntimeError(
                    f"Supabase did not return claim {index}."
                )

            claim_id = claim_response.data[0]["id"]
            selected_ids = set(item.get("citations") or [])

            # Include both all search results and specifically selected sources.
            sources_by_id = {}

            for source in (
                item.get("search_results", [])
                + item.get("sources", [])
            ):
                source_id = str(source.get("source_id") or "")
                if source_id:
                    sources_by_id[source_id] = source

            evidence_rows = []

            for source_id, source in sources_by_id.items():
                url = str(source.get("url") or "").strip()
                if not url:
                    continue

                evidence_rows.append({
                    "claim_id": claim_id,
                    "source_id": source_id,
                    "title": str(source.get("title") or "")[:300],
                    "url": url,
                    "snippet": str(source.get("snippet") or ""),
                    "score": _safe_score(source.get("score")),
                    "evidence_type": source.get("evidence_type"),
                    "authority_tier": source.get("authority_tier"),
                    "source_role": source.get("source_role"),
                    "selected_for_verdict": source_id in selected_ids,
                })

            if evidence_rows:
                (
                    client.table("fact_check_evidence")
                    .insert(evidence_rows)
                    .execute()
                )

    except Exception:
        # Avoid leaving a partial report if a child insert fails.
        try:
            (
                client.table("fact_check_reports")
                .delete()
                .eq("id", report_id)
                .execute()
            )
        except Exception:
            pass
        raise

    return report_id