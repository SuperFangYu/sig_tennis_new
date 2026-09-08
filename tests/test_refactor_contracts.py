import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from algorithm.common.action_angles import run_action_angles_csv
from algorithm.common.analysis_overlay import (
    build_halpe26_visual_points,
    render_pose_racket_video,
    stabilize_racket_visual_points,
)
from algorithm.common.manual_analysis import normalize_action
from algorithm.common.pose_features import EIGHT_ANGLE_COLUMNS, HALPE26_NAMES
from backend.app import app
from backend.routers.action_router import classify_artifact
from backend.services.pipeline import _collect_action_artifacts


class TestApiContracts(unittest.TestCase):
    def test_all_existing_action_routes_remain_registered(self) -> None:
        paths = {route.path for route in app.routes}
        for action in ("forehand", "backhand", "serve", "volley"):
            self.assertIn(f"/api/{action}/health", paths)
            self.assertIn(f"/api/{action}/analyze", paths)
            self.assertIn(f"/api/{action}/artifacts/{{run_id}}", paths)
            self.assertIn(f"/api/{action}/file", paths)

    def test_openapi_operation_ids_are_unique(self) -> None:
        operation_ids = []
        for path_item in app.openapi()["paths"].values():
            for operation in path_item.values():
                if isinstance(operation, dict) and operation.get("operationId"):
                    operation_ids.append(operation["operationId"])
        self.assertEqual(len(operation_ids), len(set(operation_ids)))


class TestArtifactContracts(unittest.TestCase):
    def test_action_specific_artifact_names(self) -> None:
        cases = (
            ("forehand", "demo_body_forehand_rtmpose.csv", "body_csv"),
            ("backhand", "demo_Backhand_Prep_v2_1.mp4", "clip"),
            ("serve", "demo_serve_trace_chart.png", "serve_trace_chart"),
            ("volley", "demo_Volley_v2_1.mp4", "clip"),
            ("volley", "demo_volley_kinetic_chart_1.png", "volley_kinetic_chart"),
        )
        for action, filename, expected in cases:
            with self.subTest(action=action, filename=filename):
                self.assertEqual(classify_artifact(action, Path(filename)), expected)

    def test_pipeline_artifact_shapes_are_preserved(self) -> None:
        point = {"csv": "racket.csv", "chart": "racket.png"}
        pose = {"csv": "body.csv"}
        common = {
            "clips": ["clip.mp4"],
            "intervals": [(1.0, 2.0)],
            "kinematic_summary_csv": "summary.csv",
        }

        forehand = _collect_action_artifacts(
            "forehand",
            point,
            pose,
            {**common, "chart": "final.png", "kinetic_charts": ["kinetic.png"]},
        )
        self.assertEqual(forehand["final_chart"], "final.png")
        self.assertEqual(forehand["kinetic_charts"], ["kinetic.png"])

        serve = _collect_action_artifacts(
            "serve",
            point,
            pose,
            {**common, "trace_chart": "trace.png", "kinetic_chart": "serve.png"},
        )
        self.assertEqual(serve["serve_trace_chart"], "trace.png")
        self.assertEqual(serve["serve_kinetic_chart"], "serve.png")

        volley = _collect_action_artifacts(
            "volley",
            point,
            pose,
            {**common, "volley_trace_charts": ["trace.png"]},
        )
        self.assertEqual(volley["volley_trace_charts"], ["trace.png"])


class TestStandaloneAngleEntrypoints(unittest.TestCase):
    def test_four_action_modules_import_without_loading_models(self) -> None:
        from algorithm.backhand.angles_csv import run_angles_csv as backhand
        from algorithm.forehand.angles_csv import run_angles_csv as forehand
        from algorithm.serve.angles_csv import run_angles_csv as serve
        from algorithm.volley.angles_csv import run_angles_csv as volley

        self.assertTrue(all(callable(func) for func in (forehand, backhand, serve, volley)))

    def test_unknown_action_fails_before_model_loading(self) -> None:
        with self.assertRaises(ValueError):
            run_action_angles_csv("unknown", "video.mp4", "output")


class TestManualAnalysisEntrypoints(unittest.TestCase):
    def test_five_pycharm_scripts_are_importable_without_loading_models(self) -> None:
        from standalone import (
            backhand_analysis,
            backhand_volley_analysis,
            forehand_analysis,
            forehand_volley_analysis,
            serve_analysis,
        )

        modules = (
            forehand_analysis,
            backhand_analysis,
            forehand_volley_analysis,
            backhand_volley_analysis,
            serve_analysis,
        )
        supported_video_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
        self.assertTrue(all(module.INPUT_VIDEO.suffix.lower() in supported_video_suffixes for module in modules))
        self.assertTrue(all(module.OUTPUT_ROOT.name == "output" for module in modules))

    def test_manual_volley_actions_are_split(self) -> None:
        self.assertEqual(normalize_action("forehand_volley"), "forehand_volley")
        self.assertEqual(normalize_action("backhand_volley"), "backhand_volley")
        with self.assertRaises(ValueError):
            normalize_action("volley")

    def test_unknown_manual_action_fails_before_model_loading(self) -> None:
        with self.assertRaises(ValueError):
            normalize_action("smash")

    def test_manual_body_csv_contract_is_exactly_eight_angles(self) -> None:
        self.assertEqual(len(EIGHT_ANGLE_COLUMNS), 8)
        self.assertEqual(
            set(EIGHT_ANGLE_COLUMNS),
            {
                "left_shoulder_angle",
                "right_shoulder_angle",
                "left_elbow_angle",
                "right_elbow_angle",
                "left_hip_angle",
                "right_hip_angle",
                "left_knee_angle",
                "right_knee_angle",
            },
        )

    def test_racket_visual_stabilizer_allows_edge_on_face_and_repairs_short_outlier(self) -> None:
        rows = []
        normal = {
            "t": (60, 20),
            "l": (59, 40),
            "r": (61, 40),  # 侧视拍面很窄，应保留
            "b": (60, 60),
            "h": (60, 90),
        }
        for frame in (1, 2, 3, 4):
            row = {"frame": frame}
            for name, (x, y) in normal.items():
                if frame == 3:
                    x += 900
                    y += 900
                row[f"{name}_x_clean"] = x
                row[f"{name}_y_clean"] = y
                row[f"{name}_conf"] = 0.9
            rows.append(row)

        stabilized = stabilize_racket_visual_points(pd.DataFrame(rows))
        self.assertEqual(stabilized[1]["l"], (59, 40))
        self.assertEqual(stabilized[1]["r"], (61, 40))
        self.assertEqual(stabilized[3]["t"], (60, 20))

    def test_halpe26_visual_head_uses_ears_and_synthetic_face_center(self) -> None:
        keypoints = np.asarray(
            [[10 + index, 20 + index] for index in range(26)], dtype=np.float32
        )
        keypoints[3] = (40, 30)
        keypoints[4] = (60, 34)
        scores = np.full(26, 0.9, dtype=np.float32)

        points = build_halpe26_visual_points((keypoints, scores), 0.3)

        self.assertNotIn("nose", points)
        self.assertNotIn("left_eye", points)
        self.assertNotIn("right_eye", points)
        self.assertNotIn("head", points)
        self.assertEqual(points["left_ear"], (40, 30))
        self.assertEqual(points["right_ear"], (60, 34))
        self.assertEqual(points["face_center"], (50, 32))
        self.assertEqual(len(points), 23)

    def test_overlay_video_renders_body_and_racket_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_video = root / "input.mp4"
            output_video = root / "overlay.mp4"
            writer = cv2.VideoWriter(
                str(input_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                25.0,
                (160, 120),
            )
            self.assertTrue(writer.isOpened())
            for _ in range(2):
                writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
            writer.release()

            body_csv = root / "body.csv"
            pd.DataFrame(
                [
                    {
                        "frame": frame,
                        "left_shoulder_x": 40,
                        "left_shoulder_y": 40,
                        "left_shoulder_conf": 0.9,
                        "right_shoulder_x": 80,
                        "right_shoulder_y": 40,
                        "right_shoulder_conf": 0.9,
                    }
                    for frame in (1, 2)
                ]
            ).to_csv(body_csv, index=False)

            racket_csv = root / "racket.csv"
            racket_rows = []
            racket_xy = {"t": (120, 30), "l": (108, 45), "r": (132, 45), "b": (120, 60), "h": (120, 88)}
            for frame in (1, 2):
                row = {"frame": frame}
                for name, (x, y) in racket_xy.items():
                    row[f"{name}_x_clean"] = x
                    row[f"{name}_y_clean"] = y
                    row[f"{name}_conf"] = 0.9
                racket_rows.append(row)
            pd.DataFrame(racket_rows).to_csv(racket_csv, index=False)

            result = render_pose_racket_video(
                input_video,
                body_csv,
                racket_csv,
                output_video,
                action="serve",
                pose_frames=[
                    (
                        np.asarray(
                            [[20 + index * 3, 20 + index * 2] for index in range(26)],
                            dtype=np.float32,
                        ),
                        np.full(26, 0.9, dtype=np.float32),
                    )
                    for _ in range(2)
                ],
            )
            self.assertEqual(Path(result), output_video.resolve())
            self.assertTrue(output_video.exists())
            self.assertGreater(output_video.stat().st_size, 0)

            cap = cv2.VideoCapture(str(output_video))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            ok, rendered = cap.read()
            cap.release()
            self.assertTrue(ok)
            self.assertGreater(int(rendered.sum()), 0)
            self.assertEqual(len(HALPE26_NAMES), 26)


class TestFrontendContracts(unittest.TestCase):
    def test_shared_analysis_page_and_legacy_redirects_exist(self) -> None:
        root = Path(__file__).resolve().parents[1] / "Vue"
        shared = (root / "analysis.html").read_text(encoding="utf-8")
        self.assertIn("createActionAnalysisApp", shared)
        for action in ("forehand", "backhand", "serve", "volley"):
            redirect = (root / f"{action}.html").read_text(encoding="utf-8")
            self.assertIn(f"/analysis.html?action={action}", redirect)


if __name__ == "__main__":
    unittest.main()
