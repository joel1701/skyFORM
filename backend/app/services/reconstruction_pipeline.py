from pathlib import Path
import shutil
import subprocess
import traceback

from app.services.frame_service import (
    select_keyframes,
    save_selected_keyframes,
)

from app.services.dynamic_object_service import (
    mask_dynamic_objects,
)

from app.services.depth_service import (
    generate_depth_maps,
)

from app.services.fusion_service import (
    fuse_depth_maps,
)

from app.services.mesh_service import (
    create_mesh_from_point_cloud,
)

from app.services.quality_service import (
    get_quality_metrics,
)

from app.services.pipeline_service import (
    mark_pipeline_started,
    mark_pipeline_complete,
    mark_pipeline_failed,
    update_pipeline_state,
)

from app.services.geospatial_service import (
    parse_gps_csv,
    get_gps_path,
)

from app.services.alignment_service import (
    georeference_reconstruction,
)


# ============================================================
# HELPERS
# ============================================================

def run_command(command):
    print()
    print("=" * 70)
    print("RUNNING:")
    print(" ".join(command))
    print("=" * 70)

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(command)
            + "\n\n"
            + result.stdout
        )

    return result.stdout


def clear_directory(path):
    path = Path(path)

    if path.exists():
        shutil.rmtree(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def find_sparse_model(sparse_dir):
    """
    Select the valid COLMAP sparse model with the largest number
    of registered images.

    This reads the image count directly from images.bin instead
    of calling `colmap model_analyzer`, which can crash natively
    on Apple Silicon.
    """
    import struct

    sparse_dir = Path(sparse_dir)

    if not sparse_dir.exists():
        raise RuntimeError(
            "COLMAP sparse directory does not exist."
        )

    valid_models = []

    for model_dir in sorted(
        sparse_dir.iterdir()
    ):
        if not model_dir.is_dir():
            continue

        cameras_file = (
            model_dir
            / "cameras.bin"
        )

        images_file = (
            model_dir
            / "images.bin"
        )

        points_file = (
            model_dir
            / "points3D.bin"
        )

        if not (
            cameras_file.exists()
            and images_file.exists()
            and points_file.exists()
        ):
            continue

        try:
            with open(
                images_file,
                "rb",
            ) as file:
                header = file.read(8)

            if len(header) != 8:
                continue

            registered_images = (
                struct.unpack(
                    "<Q",
                    header,
                )[0]
            )

            valid_models.append(
                (
                    registered_images,
                    model_dir,
                )
            )

        except Exception as error:
            print(
                "Sparse model inspection failed:",
                model_dir,
                error,
            )

    if not valid_models:
        raise RuntimeError(
            "COLMAP did not generate a valid sparse model."
        )

    valid_models.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    registered_images, best_model = (
        valid_models[0]
    )

    print(
        "Selected sparse model:",
        best_model,
        "| registered images:",
        registered_images,
    )

    return best_model


# ============================================================
# COMPLETE SKYFORM PIPELINE
# ============================================================

def run_reconstruction_pipeline(
    base_dir,
    video_path,
):
    base_dir = Path(base_dir)
    video_path = Path(video_path)
    video_id = video_path.stem

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video does not exist: {video_path}"
        )


    # --------------------------------------------------------
    # OUTPUT PATHS
    # --------------------------------------------------------

    output_dir = (
        base_dir
        / "outputs"
    )

    keyframe_dir = (
        output_dir
        / "keyframes"
        / "new_test"
    )

    colmap_keyframe_dir = (
        output_dir
        / "keyframes"
        / "colmap_input"
    )

    colmap_dir = (
        output_dir
        / "colmap_new"
    )

    sparse_dir = (
        colmap_dir
        / "sparse"
    )

    dense_dir = (
        colmap_dir
        / "dense"
    )

    dense_image_dir = (
        dense_dir
        / "images"
    )

    dense_sparse_txt_dir = (
        dense_dir
        / "sparse_txt"
    )

    depth_dir = (
        output_dir
        / "depth_maps"
    )

    reconstruction_dir = (
        output_dir
        / "reconstruction"
    )

    fused_cloud_path = (
        reconstruction_dir
        / "dense_fused.ply"
    )

    mesh_path = (
        reconstruction_dir
        / "mesh.ply"
    )

    georef_cloud_path = (
        reconstruction_dir
        / "dense_fused_georef.ply"
    )

    georef_mesh_path = (
        reconstruction_dir
        / "mesh_georef.ply"
    )

    georef_transform_path = (
    output_dir
    / "telemetry"
    / video_id
    / "geospatial_transform.json"
)


    # ========================================================
    # START PIPELINE
    # ========================================================

    mark_pipeline_started(
    base_dir,
    video_id=video_id,
)

    # Never allow geospatial artifacts from an older
    # reconstruction to be mistaken for the current run.
    for stale_path in (
        georef_cloud_path,
        georef_mesh_path,
        georef_transform_path,
    ):
        if stale_path.exists():
            stale_path.unlink()

    try:

        # ====================================================
        # STAGE 1
        # FRAME INTELLIGENCE
        # ====================================================

        update_pipeline_state(
            base_dir,
            progress=5,
            current_stage="frame_intelligence",
            message=(
                "Analyzing video and selecting "
                "reconstruction keyframes."
            ),
            stage="frame_intelligence",
            stage_status="running",
        )

        clear_directory(
            keyframe_dir
        )


        # First try stronger quality filtering.
        keyframe_result = (
            select_keyframes(
                str(video_path),
                sample_interval=5,
                sharpness_threshold=80.0,
                difference_threshold=18.0,
            )
        )


        # If too few frames survive,
        # automatically relax thresholds.
        if (
            keyframe_result[
                "selected_count"
            ]
            < 8
        ):
            print(
                "Too few keyframes selected."
            )

            print(
                "Retrying with relaxed "
                "reconstruction thresholds..."
            )

            keyframe_result = (
                select_keyframes(
                    str(video_path),
                    sample_interval=5,
                    sharpness_threshold=50.0,
                    difference_threshold=5.0,
                )
            )


        selected = (
            keyframe_result[
                "keyframes"
            ]
        )


        if len(selected) < 5:
            raise RuntimeError(
                "Not enough usable keyframes "
                "for reconstruction. "
                "Capture the object while moving "
                "around it with more viewpoint change."
            )


        saved_frames = (
            save_selected_keyframes(
                str(video_path),
                selected,
                str(keyframe_dir),
            )
        )


        if len(saved_frames) < 5:
            raise RuntimeError(
                "Failed to save enough keyframes."
            )


        print(
            "Selected keyframes:",
            len(saved_frames),
        )


        update_pipeline_state(
            base_dir,
            progress=15,
            message=(
                f"{len(saved_frames)} "
                "keyframes selected."
            ),
            stage="frame_intelligence",
            stage_status="completed",
        )


        # ====================================================
        # DYNAMIC OBJECT FILTERING
        # ====================================================

        clear_directory(
            colmap_keyframe_dir
        )

        dynamic_result = (
            mask_dynamic_objects(
                keyframe_dir,
                colmap_keyframe_dir,
            )
        )

        print(
            "Dynamic objects detected:",
            dynamic_result[
                "dynamic_objects_detected"
            ],
        )


        # ====================================================
        # STAGE 2
        # COLMAP FEATURE EXTRACTION
        # ====================================================

        update_pipeline_state(
            base_dir,
            progress=20,
            current_stage="feature_extraction",
            message=(
                "Extracting visual features "
                "with COLMAP."
            ),
            stage="feature_extraction",
            stage_status="running",
        )


        clear_directory(
            colmap_dir
        )

        sparse_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        database_path = (
            colmap_dir
            / "database.db"
        )


        run_command([
            "colmap",
            "feature_extractor",

            "--database_path",
            str(database_path),

            "--image_path",
            str(colmap_keyframe_dir),

            "--ImageReader.single_camera",
            "1",
        ])


        run_command([
            "colmap",
            "sequential_matcher",

            "--database_path",
            str(database_path),
        ])


        update_pipeline_state(
            base_dir,
            progress=32,
            message=(
                "Feature extraction and "
                "matching completed."
            ),
            stage="feature_extraction",
            stage_status="completed",
        )


        # ====================================================
        # STAGE 3
        # CAMERA POSES / SPARSE SFM
        # ====================================================

        update_pipeline_state(
            base_dir,
            progress=35,
            current_stage="camera_reconstruction",
            message=(
                "Estimating camera poses "
                "and sparse 3D geometry."
            ),
            stage="camera_reconstruction",
            stage_status="running",
        )


        run_command([
            "colmap",
            "mapper",

            "--database_path",
            str(database_path),

            "--image_path",
            str(colmap_keyframe_dir),

            "--output_path",
            str(sparse_dir),
        ])


        sparse_model = (
            find_sparse_model(
                sparse_dir
            )
        )


        print(
            "Sparse model:",
            sparse_model,
        )


        dense_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        # ====================================================
        # UNDISTORT IMAGES
        # ====================================================

        run_command([
            "colmap",
            "image_undistorter",

            "--image_path",
            str(keyframe_dir),

            "--input_path",
            str(sparse_model),

            "--output_path",
            str(dense_dir),

            "--output_type",
            "COLMAP",
        ])


        if not dense_image_dir.exists():
            raise RuntimeError(
                "COLMAP image undistortion failed."
            )


        # ====================================================
        # CONVERT COLMAP MODEL TO TXT
        # ====================================================

        dense_sparse_txt_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        dense_sparse_model = (
            dense_dir
            / "sparse"
        )


        run_command([
            "colmap",
            "model_converter",

            "--input_path",
            str(dense_sparse_model),

            "--output_path",
            str(dense_sparse_txt_dir),

            "--output_type",
            "TXT",
        ])


        required_files = [
            dense_sparse_txt_dir
            / "cameras.txt",

            dense_sparse_txt_dir
            / "images.txt",

            dense_sparse_txt_dir
            / "points3D.txt",
        ]


        for required in required_files:
            if not required.exists():
                raise RuntimeError(
                    f"Missing COLMAP file: "
                    f"{required.name}"
                )


        update_pipeline_state(
            base_dir,
            progress=48,
            message=(
                "Camera poses and sparse "
                "geometry reconstructed."
            ),
            stage="camera_reconstruction",
            stage_status="completed",
        )


        # ====================================================
        # STAGE 4
        # DEPTH ANYTHING V2
        # ====================================================

        update_pipeline_state(
            base_dir,
            progress=52,
            current_stage="depth_estimation",
            message=(
                "Generating AI relative "
                "depth maps."
            ),
            stage="depth_estimation",
            stage_status="running",
        )


        clear_directory(
            depth_dir
        )


        depth_results = (
            generate_depth_maps(
                dense_image_dir,
                depth_dir,
            )
        )


        if not depth_results:
            raise RuntimeError(
                "Depth estimation produced "
                "no depth maps."
            )


        print(
            "Depth maps generated:",
            len(depth_results),
        )


        update_pipeline_state(
            base_dir,
            progress=67,
            message=(
                f"{len(depth_results)} "
                "AI depth maps generated."
            ),
            stage="depth_estimation",
            stage_status="completed",
        )


        # ====================================================
        # STAGE 5
        # POSE-AWARE DEPTH FUSION
        # ====================================================

        update_pipeline_state(
            base_dir,
            progress=70,
            current_stage="depth_fusion",
            message=(
                "Aligning AI depth with "
                "COLMAP geometry and fusing views."
            ),
            stage="depth_fusion",
            stage_status="running",
        )


        reconstruction_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        fusion_result = (
            fuse_depth_maps(
                image_dir=dense_image_dir,
                depth_dir=depth_dir,
                model_dir=dense_sparse_txt_dir,
                output_path=fused_cloud_path,
                sample_step=12,
            )
        )


        if (
            not fused_cloud_path.exists()
        ):
            raise RuntimeError(
                "Dense fusion did not "
                "produce a point cloud."
            )


        print(
            "Views fused:",
            fusion_result["views"],
        )

        print(
            "Dense points:",
            fusion_result["points"],
        )


        update_pipeline_state(
            base_dir,
            progress=82,
            message=(
                f"{fusion_result['views']} views "
                f"fused into "
                f"{fusion_result['points']:,} points."
            ),
            stage="depth_fusion",
            stage_status="completed",
        )


        # ====================================================
        # STAGE 6
        # BALL-PIVOTING MESH
        # ====================================================

        update_pipeline_state(
            base_dir,
            progress=85,
            current_stage="mesh_generation",
            message=(
                "Generating surface mesh "
                "from fused geometry."
            ),
            stage="mesh_generation",
            stage_status="running",
        )


        if mesh_path.exists():
            mesh_path.unlink()


        mesh_result = (
            create_mesh_from_point_cloud(
                fused_cloud_path,
                mesh_path,
            )
        )


        if not mesh_path.exists():
            raise RuntimeError(
                "Mesh generation failed."
            )


        print(
            "Mesh vertices:",
            mesh_result["vertices"],
        )

        print(
            "Mesh triangles:",
            mesh_result["triangles"],
        )


        update_pipeline_state(
            base_dir,
            progress=94,
            message=(
                f"Mesh generated with "
                f"{mesh_result['vertices']:,} vertices "
                f"and "
                f"{mesh_result['triangles']:,} triangles."
            ),
            stage="mesh_generation",
            stage_status="completed",
        )


        # ====================================================
        # STAGE 7
        # QUALITY ANALYSIS
        # ====================================================

        update_pipeline_state(
            base_dir,
            progress=96,
            current_stage="quality_analysis",
            message=(
                "Calculating reconstruction "
                "confidence metrics."
            ),
            stage="quality_analysis",
            stage_status="running",
        )


        quality_result = (
            get_quality_metrics(
                base_dir
            )
        )


        if not quality_result.get(
            "available"
        ):
            raise RuntimeError(
                "Quality analysis unavailable."
            )


        summary = (
            quality_result[
                "summary"
            ]
        )


        print(
            "Quality points:",
            summary["point_count"],
        )

        print(
            "Mean heuristic confidence:",
            summary[
                "average_confidence"
            ],
        )

        print(
            "Mean reprojection error:",
            summary[
                "average_reprojection_error"
            ],
        )


        update_pipeline_state(
            base_dir,
            progress=99,
            message=(
                "Quality analysis completed."
            ),
            stage="quality_analysis",
            stage_status="completed",
        )


        # ====================================================
        # OPTIONAL GEOSPATIAL ALIGNMENT
        # ====================================================

        telemetry_file = (
            get_gps_path(
                base_dir,
                video_id,
            )
        )

        geospatial_result = None

        if telemetry_file.exists():
            try:
                gps_data = parse_gps_csv(
                    telemetry_file
                )

                gps_points = gps_data.get(
                    "points",
                    []
                )

                # Real alignment needs timestamped GPS samples.
                timestamped_points = [
                    point
                    for point in gps_points
                    if point.get("timestamp") not in (
                        None,
                        "",
                    )
                ]

                if len(timestamped_points) < 3:
                    raise RuntimeError(
                        "GPS telemetry exists, but at least "
                        "3 timestamped samples are required "
                        "for metric/geospatial alignment."
                    )

                update_pipeline_state(
                    base_dir,
                    progress=99,
                    current_stage="geospatial_alignment",
                    message=(
                        "Aligning reconstruction "
                        "to GPS telemetry."
                    ),
                )

                geospatial_result = (
                    georeference_reconstruction(
                        base_dir,
                        video_path,
                        gps_data,
                    )
                )

                print(
                    "GPS alignment complete."
                )

                print(
                    "Matched cameras:",
                    geospatial_result[
                        "matched_cameras"
                    ],
                )

                print(
                    "Alignment RMSE:",
                    geospatial_result[
                        "alignment_rmse_m"
                    ],
                    "m",
                )

            except Exception as error:
                geospatial_result = {
                    "available": False,
                    "aligned": False,
                    "reason": str(error),
                }

                print(
                    "GPS alignment skipped:",
                    error,
                )

        else:
            geospatial_result = {
                "available": False,
                "aligned": False,
                "reason": (
                    "No GPS telemetry supplied. "
                    "Reconstruction remains relative/unscaled."
                ),
            }


        # ====================================================
        # COMPLETE
        # ====================================================

        mark_pipeline_complete(
            base_dir
        )


        return {
            "success": True,

            "video":
                str(video_path),

            "keyframes":
                len(saved_frames),

            "dynamic_filtering":
                dynamic_result,

            "depth_maps":
                len(depth_results),

            "fusion":
                fusion_result,

            "mesh":
                mesh_result,

            "quality":
                summary,

            "geospatial":
                geospatial_result,

            "artifacts": {
                "dense_point_cloud":
                    str(
                        fused_cloud_path
                    ),

                "mesh":
                    str(
                        mesh_path
                    ),
            },
        }


    except Exception as error:

        print()
        print(
            "SKYFORM PIPELINE FAILED"
        )

        traceback.print_exc()


        mark_pipeline_failed(
            base_dir,
            str(error),
        )


        raise
