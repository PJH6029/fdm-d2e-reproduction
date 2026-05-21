from __future__ import annotations

from typing import Any


def expected_count_mismatches(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    code: str,
    audit_key: str,
) -> list[dict[str, Any]]:
    findings = []
    for key, raw_expected in sorted(expected.items()):
        try:
            expected_count = int(raw_expected)
        except (TypeError, ValueError):
            findings.append({"severity": "error", "code": f"{code}_invalid_expected", "audit_key": audit_key, "key": key, "expected": raw_expected})
            continue
        actual_value = actual.get(str(key))
        try:
            actual_count = int(actual_value) if actual_value is not None else None
        except (TypeError, ValueError):
            actual_count = None
        if actual_count != expected_count:
            findings.append(
                {
                    "severity": "error",
                    "code": code,
                    "audit_key": audit_key,
                    "key": str(key),
                    "expected": expected_count,
                    "actual": actual_value,
                }
            )
    return findings


def validate_d2e_only_completion_audit(
    audit: dict[str, Any] | None,
    *,
    audit_key: str,
    expected_variants: int,
    expected_by_source: dict[str, Any],
    expected_by_tier: dict[str, Any],
    require_pass: bool,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate a G003/G004 audit as a full-D2E source gate.

    Downstream D2E+aux, evaluation, and live-control claims must not infer full
    D2E coverage from goal status alone. This helper checks the terminal audit
    status plus the source/resolution-tier counts that prove 480p and original
    D2E variants were both consumed.
    """

    report = {
        "audit_key": audit_key,
        "status": None if audit is None else audit.get("status"),
        "error_count": None if audit is None else audit.get("error_count"),
        "included_recording_variants": None,
        "source_ids": {},
        "resolution_tiers": {},
        "decode_source_ids": {},
        "decode_resolution_tiers": {},
    }
    if audit is None:
        findings.append({"severity": "error", "code": "missing_d2e_only_completion_audit", "audit_key": audit_key})
        return report
    if require_pass and audit.get("status") != "pass":
        findings.append(
            {
                "severity": "error",
                "code": "d2e_only_completion_audit_not_pass",
                "audit_key": audit_key,
                "status": audit.get("status"),
                "error_count": audit.get("error_count"),
            }
        )
    universe_counts = audit.get("data_universe_counts") if isinstance(audit.get("data_universe_counts"), dict) else {}
    included = universe_counts.get("included_recording_variants")
    report["included_recording_variants"] = included
    try:
        included_count = int(included)
    except (TypeError, ValueError):
        included_count = None
    if included_count != expected_variants:
        findings.append(
            {
                "severity": "error",
                "code": "d2e_only_audit_included_variants_mismatch",
                "audit_key": audit_key,
                "expected": expected_variants,
                "actual": included,
            }
        )
    source_ids = universe_counts.get("source_ids") if isinstance(universe_counts.get("source_ids"), dict) else {}
    tiers = universe_counts.get("resolution_tiers") if isinstance(universe_counts.get("resolution_tiers"), dict) else {}
    report["source_ids"] = dict(source_ids)
    report["resolution_tiers"] = dict(tiers)
    findings.extend(expected_count_mismatches(source_ids, expected_by_source, code="d2e_only_audit_source_count_mismatch", audit_key=audit_key))
    findings.extend(expected_count_mismatches(tiers, expected_by_tier, code="d2e_only_audit_resolution_tier_count_mismatch", audit_key=audit_key))
    decode_sources = audit.get("decode_counts_by_source") if isinstance(audit.get("decode_counts_by_source"), dict) else {}
    decode_tiers = audit.get("decode_counts_by_resolution_tier") if isinstance(audit.get("decode_counts_by_resolution_tier"), dict) else {}
    report["decode_source_ids"] = dict(decode_sources)
    report["decode_resolution_tiers"] = dict(decode_tiers)
    if decode_sources:
        findings.extend(expected_count_mismatches(decode_sources, expected_by_source, code="d2e_only_audit_decode_source_count_mismatch", audit_key=audit_key))
    if decode_tiers:
        findings.extend(expected_count_mismatches(decode_tiers, expected_by_tier, code="d2e_only_audit_decode_resolution_tier_count_mismatch", audit_key=audit_key))
    return report
