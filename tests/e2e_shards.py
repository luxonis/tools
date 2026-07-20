# Pytest plugin for deterministic E2E test sharding.
#
# The default assignment is generated from the timing profile captured during
# Task 10. The workflow passes both the shard index and shard count explicitly
# so the CI topology is easy to review.
#
# To try another shard count later, regenerate/add another entry in
# E2E_SHARD_ASSIGNMENTS_BY_COUNT and keep the collection tests green.

from __future__ import annotations

import pytest

DEFAULT_E2E_SHARD_COUNT = 3

E2E_SHARD_ASSIGNMENTS_BY_COUNT: dict[int, tuple[frozenset[str], ...]] = {
    3: (
        frozenset(
            {
                "tests/test_end2end.py::test_cli_conversion[yolo26m]",
                "tests/test_end2end.py::test_cli_conversion[yolo26n-pose]",
                "tests/test_end2end.py::test_cli_conversion[yolo26n-seg]",
                "tests/test_end2end.py::test_cli_conversion[yolo26x]",
                "tests/test_end2end.py::test_cli_conversion[yoloe-v8l-seg]",
                "tests/test_end2end.py::test_cli_conversion[yolov10s]",
                "tests/test_end2end.py::test_cli_conversion[yolov11n]",
                "tests/test_end2end.py::test_cli_conversion[yolov11s]",
                "tests/test_end2end.py::test_cli_conversion[yolov12s]",
                "tests/test_end2end.py::test_cli_conversion[yolov12x]",
                "tests/test_end2end.py::test_cli_conversion[yolov5l6]",
                "tests/test_end2end.py::test_cli_conversion[yolov5lu]",
                "tests/test_end2end.py::test_cli_conversion[yolov5m6]",
                "tests/test_end2end.py::test_cli_conversion[yolov5m]",
                "tests/test_end2end.py::test_cli_conversion[yolov5mu]",
                "tests/test_end2end.py::test_cli_conversion[yolov6mr21]",
                "tests/test_end2end.py::test_cli_conversion[yolov6mr3]",
                "tests/test_end2end.py::test_cli_conversion[yolov6nr1]",
                "tests/test_end2end.py::test_cli_conversion[yolov6nr2]",
                "tests/test_end2end.py::test_cli_conversion[yolov6sr21]",
                "tests/test_end2end.py::test_cli_conversion[yolov6sr3]",
                "tests/test_end2end.py::test_cli_conversion[yolov6sr4]",
                "tests/test_end2end.py::test_cli_conversion[yolov6tr1]",
                "tests/test_end2end.py::test_cli_conversion[yolov8n-cls]",
                "tests/test_end2end.py::test_cli_conversion[yolov8n-obb]",
                "tests/test_end2end.py::test_cli_conversion[yolov8n-pose]",
                "tests/test_end2end.py::test_cli_conversion[yolov8s]",
                "tests/test_end2end.py::test_cli_conversion[yolov9c]",
                "tests/test_end2end.py::test_cli_conversion[yolov9e]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolo26n-pose]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolo26n-seg]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolov11n]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolov8n-pose]",
            }
        ),
        frozenset(
            {
                "tests/test_end2end.py::test_cli_conversion[yolo26l]",
                "tests/test_end2end.py::test_cli_conversion[yolo26n-sem]",
                "tests/test_end2end.py::test_cli_conversion[yolo26n]",
                "tests/test_end2end.py::test_cli_conversion[yolo26s]",
                "tests/test_end2end.py::test_cli_conversion[yoloe-11s-seg]",
                "tests/test_end2end.py::test_cli_conversion[yoloe-v8m-seg]",
                "tests/test_end2end.py::test_cli_conversion[yoloe-v8s-seg]",
                "tests/test_end2end.py::test_cli_conversion[yolov10b]",
                "tests/test_end2end.py::test_cli_conversion[yolov10m]",
                "tests/test_end2end.py::test_cli_conversion[yolov10n]",
                "tests/test_end2end.py::test_cli_conversion[yolov11m]",
                "tests/test_end2end.py::test_cli_conversion[yolov11n-seg]",
                "tests/test_end2end.py::test_cli_conversion[yolov12l]",
                "tests/test_end2end.py::test_cli_conversion[yolov26_nms]",
                "tests/test_end2end.py::test_cli_conversion[yolov5m6u]",
                "tests/test_end2end.py::test_cli_conversion[yolov5n6]",
                "tests/test_end2end.py::test_cli_conversion[yolov5n]",
                "tests/test_end2end.py::test_cli_conversion[yolov5nu]",
                "tests/test_end2end.py::test_cli_conversion[yolov5s6u]",
                "tests/test_end2end.py::test_cli_conversion[yolov5x]",
                "tests/test_end2end.py::test_cli_conversion[yolov6lr21]",
                "tests/test_end2end.py::test_cli_conversion[yolov6lr2]",
                "tests/test_end2end.py::test_cli_conversion[yolov6lr4]",
                "tests/test_end2end.py::test_cli_conversion[yolov6mr2]",
                "tests/test_end2end.py::test_cli_conversion[yolov6nr3]",
                "tests/test_end2end.py::test_cli_conversion[yolov6nr4]",
                "tests/test_end2end.py::test_cli_conversion[yolov7]",
                "tests/test_end2end.py::test_cli_conversion[yolov8m]",
                "tests/test_end2end.py::test_cli_conversion[yolov8n]",
                "tests/test_end2end.py::test_cli_conversion[yolov8x]",
                "tests/test_end2end.py::test_cli_conversion[yolov9s]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolo26n]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolov11n-seg]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolov8n]",
                "tests/test_end2end.py::test_yolo26_semseg_nnarchive_head",
            }
        ),
        frozenset(
            {
                "tests/test_end2end.py::test_cli_conversion[yoloe-11l-seg]",
                "tests/test_end2end.py::test_cli_conversion[yoloe-11m-seg]",
                "tests/test_end2end.py::test_cli_conversion[yolov10l]",
                "tests/test_end2end.py::test_cli_conversion[yolov10x]",
                "tests/test_end2end.py::test_cli_conversion[yolov11l]",
                "tests/test_end2end.py::test_cli_conversion[yolov11n-cls]",
                "tests/test_end2end.py::test_cli_conversion[yolov11n-obb]",
                "tests/test_end2end.py::test_cli_conversion[yolov11n-pose]",
                "tests/test_end2end.py::test_cli_conversion[yolov11x]",
                "tests/test_end2end.py::test_cli_conversion[yolov12m]",
                "tests/test_end2end.py::test_cli_conversion[yolov12n]",
                "tests/test_end2end.py::test_cli_conversion[yolov5l6u]",
                "tests/test_end2end.py::test_cli_conversion[yolov5l]",
                "tests/test_end2end.py::test_cli_conversion[yolov5n6u]",
                "tests/test_end2end.py::test_cli_conversion[yolov5s6]",
                "tests/test_end2end.py::test_cli_conversion[yolov5s]",
                "tests/test_end2end.py::test_cli_conversion[yolov5su]",
                "tests/test_end2end.py::test_cli_conversion[yolov6lr3]",
                "tests/test_end2end.py::test_cli_conversion[yolov6mr4]",
                "tests/test_end2end.py::test_cli_conversion[yolov6nr21]",
                "tests/test_end2end.py::test_cli_conversion[yolov6sr1]",
                "tests/test_end2end.py::test_cli_conversion[yolov6sr2]",
                "tests/test_end2end.py::test_cli_conversion[yolov6tr2]",
                "tests/test_end2end.py::test_cli_conversion[yolov7t]",
                "tests/test_end2end.py::test_cli_conversion[yolov7x]",
                "tests/test_end2end.py::test_cli_conversion[yolov8l]",
                "tests/test_end2end.py::test_cli_conversion[yolov8n-seg]",
                "tests/test_end2end.py::test_cli_conversion[yolov9m]",
                "tests/test_end2end.py::test_cli_conversion[yolov9t]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolov11n-pose]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolov12n]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolov8n-seg]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolov9t]",
            }
        ),
    ),
    10: (
        frozenset(
            {
                "tests/test_end2end.py::test_cli_conversion[yolo26n-sem]",
                "tests/test_end2end.py::test_cli_conversion[yoloe-11l-seg]",
                "tests/test_end2end.py::test_cli_conversion[yolov11s]",
                "tests/test_end2end.py::test_cli_conversion[yolov12n]",
                "tests/test_end2end.py::test_cli_conversion[yolov6lr2]",
                "tests/test_end2end.py::test_cli_conversion[yolov6nr3]",
                "tests/test_end2end.py::test_cli_conversion[yolov6tr1]",
                "tests/test_end2end.py::test_cli_conversion[yolov8l]",
                "tests/test_end2end.py::test_cli_conversion[yolov9m]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolov12n]",
                "tests/test_end2end.py::test_yolo26_semseg_nnarchive_head",
            }
        ),
        frozenset(
            {
                "tests/test_end2end.py::test_cli_conversion[yolov10m]",
                "tests/test_end2end.py::test_cli_conversion[yolov12l]",
                "tests/test_end2end.py::test_cli_conversion[yolov12m]",
                "tests/test_end2end.py::test_cli_conversion[yolov5l6]",
                "tests/test_end2end.py::test_cli_conversion[yolov6mr4]",
                "tests/test_end2end.py::test_cli_conversion[yolov6nr21]",
                "tests/test_end2end.py::test_cli_conversion[yolov8n-pose]",
                "tests/test_end2end.py::test_cli_conversion[yolov8s]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolov8n-pose]",
            }
        ),
        frozenset(
            {
                "tests/test_end2end.py::test_cli_conversion[yolo26n-pose]",
                "tests/test_end2end.py::test_cli_conversion[yolov10l]",
                "tests/test_end2end.py::test_cli_conversion[yolov11n-cls]",
                "tests/test_end2end.py::test_cli_conversion[yolov12s]",
                "tests/test_end2end.py::test_cli_conversion[yolov5l6u]",
                "tests/test_end2end.py::test_cli_conversion[yolov5m6u]",
                "tests/test_end2end.py::test_cli_conversion[yolov6mr2]",
                "tests/test_end2end.py::test_cli_conversion[yolov6sr1]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolo26n-pose]",
            }
        ),
        frozenset(
            {
                "tests/test_end2end.py::test_cli_conversion[yoloe-v8m-seg]",
                "tests/test_end2end.py::test_cli_conversion[yoloe-v8s-seg]",
                "tests/test_end2end.py::test_cli_conversion[yolov10n]",
                "tests/test_end2end.py::test_cli_conversion[yolov5l]",
                "tests/test_end2end.py::test_cli_conversion[yolov5n6]",
                "tests/test_end2end.py::test_cli_conversion[yolov6lr3]",
                "tests/test_end2end.py::test_cli_conversion[yolov6sr3]",
                "tests/test_end2end.py::test_cli_conversion[yolov8m]",
                "tests/test_end2end.py::test_cli_conversion[yolov9s]",
                "tests/test_end2end.py::test_cli_conversion[yolov9t]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolov9t]",
            }
        ),
        frozenset(
            {
                "tests/test_end2end.py::test_cli_conversion[yolo26n-seg]",
                "tests/test_end2end.py::test_cli_conversion[yolov11m]",
                "tests/test_end2end.py::test_cli_conversion[yolov5m6]",
                "tests/test_end2end.py::test_cli_conversion[yolov5mu]",
                "tests/test_end2end.py::test_cli_conversion[yolov5nu]",
                "tests/test_end2end.py::test_cli_conversion[yolov5s6u]",
                "tests/test_end2end.py::test_cli_conversion[yolov5su]",
                "tests/test_end2end.py::test_cli_conversion[yolov7x]",
                "tests/test_end2end.py::test_cli_conversion[yolov8n]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolo26n-seg]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolov8n]",
            }
        ),
        frozenset(
            {
                "tests/test_end2end.py::test_cli_conversion[yolo26l]",
                "tests/test_end2end.py::test_cli_conversion[yolo26x]",
                "tests/test_end2end.py::test_cli_conversion[yolov10b]",
                "tests/test_end2end.py::test_cli_conversion[yolov11l]",
                "tests/test_end2end.py::test_cli_conversion[yolov5n]",
                "tests/test_end2end.py::test_cli_conversion[yolov6nr1]",
                "tests/test_end2end.py::test_cli_conversion[yolov6sr21]",
                "tests/test_end2end.py::test_cli_conversion[yolov8n-seg]",
                "tests/test_end2end.py::test_cli_conversion[yolov9e]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolov8n-seg]",
            }
        ),
        frozenset(
            {
                "tests/test_end2end.py::test_cli_conversion[yoloe-11s-seg]",
                "tests/test_end2end.py::test_cli_conversion[yolov10x]",
                "tests/test_end2end.py::test_cli_conversion[yolov5lu]",
                "tests/test_end2end.py::test_cli_conversion[yolov5m]",
                "tests/test_end2end.py::test_cli_conversion[yolov5s6]",
                "tests/test_end2end.py::test_cli_conversion[yolov6nr2]",
                "tests/test_end2end.py::test_cli_conversion[yolov6nr4]",
                "tests/test_end2end.py::test_cli_conversion[yolov7t]",
                "tests/test_end2end.py::test_cli_conversion[yolov8x]",
            }
        ),
        frozenset(
            {
                "tests/test_end2end.py::test_cli_conversion[yolo26s]",
                "tests/test_end2end.py::test_cli_conversion[yoloe-11m-seg]",
                "tests/test_end2end.py::test_cli_conversion[yolov11n-pose]",
                "tests/test_end2end.py::test_cli_conversion[yolov11n-seg]",
                "tests/test_end2end.py::test_cli_conversion[yolov12x]",
                "tests/test_end2end.py::test_cli_conversion[yolov6lr21]",
                "tests/test_end2end.py::test_cli_conversion[yolov6mr3]",
                "tests/test_end2end.py::test_cli_conversion[yolov6sr2]",
                "tests/test_end2end.py::test_cli_conversion[yolov8n-cls]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolov11n-pose]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolov11n-seg]",
            }
        ),
        frozenset(
            {
                "tests/test_end2end.py::test_cli_conversion[yolo26m]",
                "tests/test_end2end.py::test_cli_conversion[yolo26n]",
                "tests/test_end2end.py::test_cli_conversion[yoloe-v8l-seg]",
                "tests/test_end2end.py::test_cli_conversion[yolov11n]",
                "tests/test_end2end.py::test_cli_conversion[yolov26_nms]",
                "tests/test_end2end.py::test_cli_conversion[yolov5x]",
                "tests/test_end2end.py::test_cli_conversion[yolov6mr21]",
                "tests/test_end2end.py::test_cli_conversion[yolov6tr2]",
                "tests/test_end2end.py::test_cli_conversion[yolov8n-obb]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolo26n]",
                "tests/test_end2end.py::test_n_variant_nnarchive_outputs[yolov11n]",
            }
        ),
        frozenset(
            {
                "tests/test_end2end.py::test_cli_conversion[yolov10s]",
                "tests/test_end2end.py::test_cli_conversion[yolov11n-obb]",
                "tests/test_end2end.py::test_cli_conversion[yolov11x]",
                "tests/test_end2end.py::test_cli_conversion[yolov5n6u]",
                "tests/test_end2end.py::test_cli_conversion[yolov5s]",
                "tests/test_end2end.py::test_cli_conversion[yolov6lr4]",
                "tests/test_end2end.py::test_cli_conversion[yolov6sr4]",
                "tests/test_end2end.py::test_cli_conversion[yolov7]",
                "tests/test_end2end.py::test_cli_conversion[yolov9c]",
            }
        ),
    ),
}


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("e2e-shards")
    group.addoption(
        "--e2e-shard-index",
        type=int,
        default=None,
        help=(
            "Run only one deterministic public E2E shard. "
            "Use together with --e2e-shard-count."
        ),
    )
    group.addoption(
        "--e2e-shard-count",
        type=int,
        default=DEFAULT_E2E_SHARD_COUNT,
        help=(
            "Total number of public E2E shards. "
            "Currently supported counts are listed in "
            "E2E_SHARD_ASSIGNMENTS_BY_COUNT."
        ),
    )


def supported_e2e_shard_counts() -> tuple[int, ...]:
    return tuple(sorted(E2E_SHARD_ASSIGNMENTS_BY_COUNT))


def get_e2e_shard_assignment(shard_count: int) -> tuple[frozenset[str], ...]:
    try:
        return E2E_SHARD_ASSIGNMENTS_BY_COUNT[shard_count]
    except KeyError as exc:
        supported = ", ".join(str(count) for count in supported_e2e_shard_counts())
        raise pytest.UsageError(
            f"No E2E shard assignment exists for count={shard_count}. "
            f"Supported counts: {supported}. "
            "Regenerate the shard assignment from the timing profile before "
            "using this count."
        ) from exc


def get_e2e_shard_items(shard_index: int, shard_count: int) -> frozenset[str]:
    assignment = get_e2e_shard_assignment(shard_count)
    if shard_index < 0 or shard_index >= shard_count:
        raise pytest.UsageError(
            f"Invalid E2E shard index {shard_index} for count={shard_count}."
        )
    if len(assignment) != shard_count:
        raise pytest.UsageError(
            f"E2E shard assignment for count={shard_count} has "
            f"{len(assignment)} shards."
        )

    return assignment[shard_index]


def validate_e2e_shard_assignment(
    public_nodeids: set[str],
    assignment: tuple[frozenset[str], ...],
) -> None:
    manifest_nodeids = set().union(*assignment)

    missing_from_assignment = sorted(public_nodeids - manifest_nodeids)
    stale_in_assignment = sorted(manifest_nodeids - public_nodeids)

    if missing_from_assignment or stale_in_assignment:
        raise pytest.UsageError(
            "Public E2E collection changed. Regenerate "
            "E2E_SHARD_ASSIGNMENTS_BY_COUNT. "
            f"Missing from assignment: {missing_from_assignment}. "
            f"Stale in assignment: {stale_in_assignment}."
        )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    shard_index = config.getoption("e2e_shard_index")
    if shard_index is None:
        return

    shard_count = config.getoption("e2e_shard_count")
    selected_nodeids = get_e2e_shard_items(shard_index, shard_count)
    assignment = get_e2e_shard_assignment(shard_count)

    public_nodeids = {
        item.nodeid
        for item in items
        if item.nodeid.startswith("tests/test_end2end.py::")
        and "::test_private_model_conversion" not in item.nodeid
    }

    validate_e2e_shard_assignment(
        public_nodeids,
        assignment,
    )

    selected_items: list[pytest.Item] = []
    deselected_items: list[pytest.Item] = []

    for item in items:
        if item.nodeid in selected_nodeids:
            selected_items.append(item)
        else:
            deselected_items.append(item)

    if deselected_items:
        config.hook.pytest_deselected(items=deselected_items)

    items[:] = selected_items
