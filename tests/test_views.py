from __future__ import annotations

import unittest

from service.views import (
    leaderboard_page,
    problem_page,
    register_page,
    registered_page,
    submission_page,
    submit_page,
)


class ProblemPageTests(unittest.TestCase):
    def test_shared_navigation_links_to_problem_page(self) -> None:
        self.assertIn('<a href="/problem">Problem</a>', leaderboard_page([]))

    def test_page_explains_recurrence_prompt_and_exact_scoring(self) -> None:
        page = problem_page()
        self.assertIn("The benchmark problem", page)
        self.assertIn("x</var><sub>t−1</sub><sup>2</sup> mod <var>N</var>", page)
        self.assertIn("N = 77 · x = 2 · T = 4", page)
        self.assertIn("25² mod 77", page)
        self.assertIn("Its secret prime factors", page)
        self.assertIn("every target token is correct", page)
        self.assertIn("Evaluator-only shortcut", page)

    def test_problem_route_is_registered(self) -> None:
        from service.app import app

        problem_routes = [
            route for route in app.routes if getattr(route, "path", None) == "/problem"
        ]
        self.assertEqual(len(problem_routes), 1)
        self.assertIn("GET", problem_routes[0].methods)


class RegisterPageTests(unittest.TestCase):
    def test_page_links_to_github_authentication(self) -> None:
        page = register_page()
        self.assertIn('href="/auth/github"', page)
        self.assertIn("Continue with GitHub", page)
        self.assertNotIn("Luma", page)

    def test_registered_page_escapes_identity_and_shows_key(self) -> None:
        page = registered_page(
            {"display_name": "Ada <script>", "github_login": "ada"},
            "old_example",
        )
        self.assertIn("Ada &lt;script&gt;", page)
        self.assertNotIn("Ada <script>", page)
        self.assertIn("old_example", page)

class SubmissionRulesPageTests(unittest.TestCase):
    def test_explicit_submission_rules_are_rendered(self) -> None:
        page = submit_page()
        self.assertIn('id="submission-rules-heading">Rules</h2>', page)
        self.assertIn("Maximum 500 million trainable parameters", page)
        self.assertIn("No hard-coded weights", page)
        self.assertIn("use a random initialization", page)
        self.assertIn("<code>torch.load</code> is not allowed", page)
        self.assertIn("No hard-coded algorithm in the forward pass", page)
        self.assertIn("End-to-end learning only", page)
        self.assertIn(
            "with all input-dependent computation inside the autograd graph",
            page,
        )
        self.assertIn(
            "an unbroken gradient path from the loss to the parameters responsible for the prediction",
            page,
        )
        self.assertIn("Everything stays on the GPU", page)
        self.assertIn("CPU offloading is not allowed", page)
        self.assertIn("Repeated rule-breaking will get you banned", page)
        self.assertIn("The metric recorder for a Hard run must not be exploited", page)
        self.assertIn("Any attempt to exploit it will result in an immediate ban", page)
        self.assertIn("discussing possible loopholes on Discord", page)
        self.assertIn("testing one in a submission won't get you banned", page)
        self.assertLess(
            page.index("<strong>Evaluator boundary:</strong>"),
            page.index('id="submission-rules-heading">Rules</h2>'),
        )
        self.assertIn(
            "<span>05</span><p><strong>Everything stays on the GPU.",
            page,
        )
        self.assertIn(
            "<span>06</span><p><strong>Repeated rule-breaking will get you banned.",
            page,
        )
        self.assertIn(
            "<span>07</span><p><strong>The metric recorder for a Hard run must not be exploited.",
            page,
        )

    def test_beta_notice_invites_feedback_before_rules_are_finalized(self) -> None:
        page = submit_page()
        self.assertIn("Beta:", page)
        self.assertIn("before we finalize the rules", page)
        self.assertIn('href="https://discord.gg/gpumode"', page)
        self.assertIn("#one-layer-deeper", page)
        self.assertNotIn("issue on GitHub", page)

    def test_compute_tier_and_dataset_selectors_are_rendered(self) -> None:
        page = submit_page()
        self.assertIn('name="tier"', page)
        self.assertIn('name="dataset"', page)
        self.assertIn("E1 · Fixed N=323", page)
        self.assertIn("Easy 1 min · Medium 10 min · Hard 1 hour", page)
        self.assertIn("It is not graded separately", page)
        self.assertIn("training_time_seconds", page)
        self.assertIn("Training loss", page)
        self.assertIn("training_loss", page)
        self.assertIn('href="/samples/submission.py"', page)
        self.assertNotIn("recurrent_adamw", page)
        self.assertNotIn("recurrent_lion", page)
        self.assertNotIn("recurrent_muon", page)
        self.assertNotIn("evaluate 1, 2, 4", page)
        self.assertNotIn("set_depth_budget", page)

    def test_hard_selection_hides_dataset_control(self) -> None:
        page = submit_page(selected_tier="hard")
        self.assertIn('value="hard" selected', page)
        self.assertIn('id="dataset-field" class="text-field" hidden', page)
        self.assertIn('id="dataset-select" name="dataset" disabled', page)
        self.assertIn("H1 · Hidden evaluation", page)


class FooterAcknowledgementTests(unittest.TestCase):
    def test_shared_footer_thanks_infrastructure_supporters(self) -> None:
        page = leaderboard_page([])
        self.assertIn('href="https://modal.com/"', page)
        self.assertIn('href="https://northflank.com/"', page)
        self.assertIn("With thanks to", page)

    def test_shared_footer_links_to_updates_channel(self) -> None:
        page = leaderboard_page([])
        self.assertIn('href="https://discord.gg/gpumode"', page)
        self.assertIn("#one-layer-deeper", page)


class CompetitionDeadlineTests(unittest.TestCase):
    def test_shared_layout_shows_submission_deadline(self) -> None:
        page = leaderboard_page([])
        self.assertIn(
            "<strong>Submission deadline:</strong> August 31 at 10:00 PM PT.",
            page,
        )
        self.assertIn('aria-label="Competition submission deadline"', page)


class CompetitionRulesLinkTests(unittest.TestCase):
    def test_leaderboard_links_to_repository(self) -> None:
        page = leaderboard_page([])
        self.assertIn(
            'href="https://github.com/tilde-research/one-layer-deeper"',
            page,
        )
        self.assertIn("Detailed rules &amp; criteria on GitHub", page)


class FaviconTests(unittest.TestCase):
    def test_shared_layout_uses_square_brand_mark_as_favicon(self) -> None:
        page = leaderboard_page([])
        self.assertIn(
            '<link rel="icon" href="/static/favicon.svg" type="image/svg+xml">',
            page,
        )


class CompetitionPrivacyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {
            "id": "00000000-0000-0000-0000-000000000001",
            "filename": "submission.py",
            "status": "succeeded",
            "created_at": "2026-07-13T12:00:00Z",
            "finished_at": "2026-07-13T12:10:00Z",
            "manifest_name": "h100_hard_h1.json",
            "tier": "hard",
            "dataset_id": "h1",
            "dataset_label": "H1 · Hidden evaluation",
            "submitter": "Ada Lovelace",
            "score": 0.75,
            # Private fields must never be rendered even if a caller supplies them.
            "name": "secret architecture name",
            "description": "secret architecture description",
            "modal_call_id": "secret-modal-call",
            "error": "secret failure details",
            "log_tail": "secret evaluator logs",
            "result": {"seeds": [{"secret": "private-seed-payload"}]},
        }

    def test_leaderboard_shows_identity_file_and_score_only(self) -> None:
        page = leaderboard_page([self.row])
        self.assertIn("Ada Lovelace", page)
        self.assertIn("submission.py", page)
        self.assertNotIn("<th>Status</th>", page)
        self.assertIn("75.00%", page)
        self.assertIn("source and run details remain private", page)
        self.assertNotIn("secret architecture", page)
        self.assertNotIn("secret-modal-call", page)

    def test_public_submission_page_hides_run_internals(self) -> None:
        page = submission_page(self.row)
        self.assertIn("Ada Lovelace", page)
        self.assertIn("submission.py", page)
        self.assertIn("Private during competition", page)
        for private_value in (
            "secret architecture name",
            "secret architecture description",
            "secret-modal-call",
            "secret failure details",
            "secret evaluator logs",
            "private-seed-payload",
        ):
            self.assertNotIn(private_value, page)


if __name__ == "__main__":
    unittest.main()
