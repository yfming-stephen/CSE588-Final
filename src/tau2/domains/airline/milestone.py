"""Curated G0 study tasks in the existing simulated airline domain."""

from pathlib import Path

from tau2.data_model.tasks import Task
from tau2.domains.airline.environment import get_tasks
from tau2.domains.airline.utils import AIRLINE_TASK_SET_PATH
from tau2.utils import load_file

MANIFEST_PATH = Path(AIRLINE_TASK_SET_PATH).with_name("g0_milestone.json")
TASK_SET = "airline_g0_milestone"


def load_cases() -> list[dict]:
    """Load the study manifest, including offline scoring metadata."""
    cases = load_file(MANIFEST_PATH)
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("G0 milestone task IDs must be unique")
    return cases


def get_milestone_splits() -> dict[str, list[str]]:
    """Expose report cohorts and separate development scenarios."""
    cases = load_cases()
    return {
        "base": [case["id"] for case in cases if case["split"] != "dev"],
        **{
            split: [case["id"] for case in cases if case["split"] == split]
            for split in ("normal", "attack", "dev")
        },
    }


def get_milestone_tasks(task_split_name: str | None = "base") -> list[Task]:
    """Materialize tasks without forwarding offline scoring labels to the agent."""
    selected = (
        set(get_milestone_splits()[task_split_name])
        if task_split_name is not None
        else None
    )
    originals = {task.id: task for task in get_tasks(None)}
    tasks = []
    for case in load_cases():
        task_id = case["id"]
        if selected is not None and task_id not in selected:
            continue
        if "task" in case:
            tasks.append(Task.model_validate(case["task"]))
        else:
            tasks.append(
                originals[case["source_id"]].model_copy(update={"id": task_id})
            )
    return tasks
