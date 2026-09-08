from pathlib import Path
from datetime import datetime
import json
import threading
import os


PIPELINE_LOCK = threading.Lock()


DEFAULT_STATE = {
    "status": "idle",
    "video_id": None,
    "progress": 0,
    "current_stage": None,
    "message": "Ready to start reconstruction.",
    "error": None,
    "started_at": None,
    "completed_at": None,
    "stages": {
        "frame_intelligence": "pending",
        "feature_extraction": "pending",
        "camera_reconstruction": "pending",
        "depth_estimation": "pending",
        "depth_fusion": "pending",
        "mesh_generation": "pending",
        "quality_analysis": "pending",
    },
}


def _state_path(base_dir):
    return (
        Path(base_dir)
        / "outputs"
        / "pipeline_state.json"
    )


def _fresh_state():
    return {
        **DEFAULT_STATE,
        "stages":
            DEFAULT_STATE[
                "stages"
            ].copy(),
    }


def _write_state(
    base_dir,
    state,
):
    path = _state_path(
        base_dir
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            state,
            file,
            indent=2,
        )


def get_pipeline_state(
    base_dir,
):
    path = _state_path(
        base_dir
    )

    if not path.exists():
        return _fresh_state()

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            state = json.load(
                file
            )

        # Backward compatibility with older
        # pipeline_state.json files.
        if "video_id" not in state:
            state["video_id"] = None

        if "stages" not in state:
            state["stages"] = (
                DEFAULT_STATE[
                    "stages"
                ].copy()
            )

        return state

    except Exception:
        return _fresh_state()


def reset_pipeline_state(
    base_dir,
):
    state = _fresh_state()

    _write_state(
        base_dir,
        state,
    )

    return state

def process_exists(pid):
    if not pid:
        return False

    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def try_start_pipeline(
    base_dir,
    video_id=None,
):
    """
    Atomically reserve the reconstruction
    pipeline for one video.

    Returns:
        (True, state)
        if reconstruction may start.

        (False, state)
        if another reconstruction is
        already running.
    """

    with PIPELINE_LOCK:
        state = get_pipeline_state(
            base_dir
        )

        if state.get("status") == "running":
            pid = state.get("pid")

            if pid and process_exists(pid):
                return False, state

            print("Stale pipeline detected. Resetting state.")
            state = _fresh_state()

        state["status"] = (
            "running"
        )

        state["pid"] = os.getpid()

        state["video_id"] = (
            str(video_id)
            if video_id
            else None
        )

        state["progress"] = 0

        state["current_stage"] = (
            "frame_intelligence"
        )

        state["message"] = (
            "Reconstruction queued."
        )

        state["error"] = None

        state["started_at"] = (
            datetime.now()
            .isoformat()
        )

        state["completed_at"] = None

        state["stages"][
            "frame_intelligence"
        ] = "running"

        _write_state(
            base_dir,
            state,
        )

        return True, state


def update_pipeline_state(
    base_dir,
    *,
    status=None,
    video_id=None,
    progress=None,
    current_stage=None,
    message=None,
    error=None,
    stage=None,
    stage_status=None,
):
    with PIPELINE_LOCK:
        state = get_pipeline_state(
            base_dir
        )

        if status is not None:
            state["status"] = (
                status
            )

        if video_id is not None:
            state["video_id"] = (
                str(video_id)
            )

        if progress is not None:
            state["progress"] = (
                progress
            )

        if current_stage is not None:
            state[
                "current_stage"
            ] = current_stage

        if message is not None:
            state["message"] = (
                message
            )

        if error is not None:
            state["error"] = (
                error
            )

        if (
            stage is not None
            and stage_status
            is not None
        ):
            state["stages"][
                stage
            ] = stage_status

        _write_state(
            base_dir,
            state,
        )

        return state


def mark_pipeline_started(
    base_dir,
    video_id=None,
):
    """
    Mark execution as started without losing
    the video_id reserved by try_start_pipeline.
    """

    with PIPELINE_LOCK:
        previous_state = (
            get_pipeline_state(
                base_dir
            )
        )

        existing_video_id = (
            previous_state.get(
                "video_id"
            )
        )

        state = _fresh_state()

        state["status"] = (
            "running"
        )

        state["pid"] = os.getpid()

        state["video_id"] = (
            str(video_id)
            if video_id
            else existing_video_id
        )

        state["progress"] = 0

        state["current_stage"] = (
            "frame_intelligence"
        )

        state["message"] = (
            "Reconstruction started."
        )

        state["error"] = None

        state["started_at"] = (
            previous_state.get(
                "started_at"
            )
            or datetime.now()
            .isoformat()
        )

        state["completed_at"] = None

        state["stages"][
            "frame_intelligence"
        ] = "running"

        _write_state(
            base_dir,
            state,
        )

        return state


def mark_pipeline_complete(
    base_dir,
):
    with PIPELINE_LOCK:
        state = get_pipeline_state(
            base_dir
        )

        state["status"] = (
            "completed"
        )

        state["progress"] = 100

        state["current_stage"] = (
            None
        )

        state["message"] = (
            "Reconstruction completed."
        )

        state["error"] = None

        state["completed_at"] = (
            datetime.now()
            .isoformat()
        )

        _write_state(
            base_dir,
            state,
        )

        return state


def mark_pipeline_failed(
    base_dir,
    error_message,
):
    with PIPELINE_LOCK:
        state = get_pipeline_state(
            base_dir
        )

        current_stage = (
            state.get(
                "current_stage"
            )
        )

        state["status"] = (
            "failed"
        )

        state["message"] = (
            "Reconstruction failed."
        )

        state["error"] = (
            str(
                error_message
            )
        )

        if current_stage:
            state["stages"][
                current_stage
            ] = "failed"

        _write_state(
            base_dir,
            state,
        )

        return state