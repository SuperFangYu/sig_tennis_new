import unittest
from pathlib import Path

from algorithm.common.action_angles import run_action_angles_csv
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
