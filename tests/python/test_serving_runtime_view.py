#  Copyright (c) 2026, TensorCast Team.

import pytest

from tensorcast.serving.runtime_view import aggregate_runtime_view_outputs


def test_runtime_view_aggregate_reports_partial_publication():
    payload = aggregate_runtime_view_outputs(
        [
            {
                "serving_artifact_ref": "mi2:serving",
                "published_replica": {"state": "published"},
            },
            {"serving_artifact_ref": "mi2:serving"},
        ],
        response_name="weight_version",
    )

    assert payload is not None
    assert payload["publication_aggregate"] == {
        "schema_version": 1,
        "state": "partial",
        "mode": "runtime_view",
        "published_workers": 1,
        "required_workers": 2,
        "failed_workers": 0,
        "pending_workers": 0,
        "stale_workers": 0,
    }


def test_runtime_view_aggregate_reports_failed_before_published():
    payload = aggregate_runtime_view_outputs(
        [
            {"published_replica": {"state": "published"}},
            {"published_replica": {"state": "failed"}},
        ],
        response_name="weight_version",
    )

    assert payload is not None
    assert payload["publication_aggregate"]["state"] == "failed"


def test_runtime_view_aggregate_reports_stale_before_partial():
    payload = aggregate_runtime_view_outputs(
        [
            {"published_replica": {"state": "published"}},
            {"published_replica": {"state": "stale"}},
            {"published_replica": {"state": "unpublished"}},
        ],
        response_name="weight_version",
    )

    assert payload is not None
    assert payload["publication_aggregate"] == {
        "schema_version": 1,
        "state": "stale",
        "mode": "runtime_view",
        "published_workers": 1,
        "required_workers": 3,
        "failed_workers": 0,
        "pending_workers": 0,
        "stale_workers": 1,
    }


def test_runtime_view_aggregate_reports_retired_publication_as_unpublished():
    payload = aggregate_runtime_view_outputs(
        [
            {"published_replica": {"state": "retired"}},
            {"published_replica": {"state": "retired"}},
        ],
        response_name="weight_version",
    )

    assert payload is not None
    assert payload["publication_aggregate"] == {
        "schema_version": 1,
        "state": "unpublished",
        "mode": "runtime_view",
        "published_workers": 0,
        "required_workers": 2,
        "failed_workers": 0,
        "pending_workers": 0,
        "stale_workers": 0,
    }


def test_runtime_view_aggregate_omits_publication_when_no_worker_reports_it():
    payload = aggregate_runtime_view_outputs(
        [
            {"serving_artifact_ref": "mi2:serving"},
            {"serving_artifact_ref": "mi2:serving"},
        ],
        response_name="weight_version",
    )

    assert payload == {"serving_artifact_ref": "mi2:serving"}


def test_runtime_view_aggregate_rejects_non_dict_worker_payload():
    with pytest.raises(RuntimeError, match="worker response must be a dict"):
        aggregate_runtime_view_outputs(
            [{"serving_artifact_ref": "mi2:serving"}, "bad"],
            response_name="weight_version",
        )
