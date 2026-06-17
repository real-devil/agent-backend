import unittest

from agents.routing import detect_intents, fallback_plan_for_intents, reconcile_plan_steps


class RoutingTests(unittest.TestCase):
    def test_weather_with_document_selected_uses_tool_not_rag(self):
        intents = detect_intents("今天北京的天气怎么样？", has_document=True)
        self.assertTrue(intents.needs_weather_tool)
        self.assertFalse(intents.needs_document_rag)

        plan = fallback_plan_for_intents("今天北京的天气怎么样？", has_document=True)
        self.assertEqual(len(plan["steps"]), 1)
        self.assertEqual(plan["steps"][0]["agent"], "tool_agent")

    def test_document_question_with_document_selected_uses_rag(self):
        intents = detect_intents("根据文档总结主要内容", has_document=True)
        self.assertFalse(intents.needs_weather_tool)
        self.assertTrue(intents.needs_document_rag)

        plan = fallback_plan_for_intents("根据文档总结主要内容", has_document=True)
        self.assertEqual(plan["steps"][0]["agent"], "rag_agent")

    def test_mixed_document_and_weather_uses_parallel_agents(self):
        message = "根据文档总结要点，并查询今天北京天气"
        intents = detect_intents(message, has_document=True)
        self.assertTrue(intents.needs_weather_tool)
        self.assertTrue(intents.needs_document_rag)

        plan = fallback_plan_for_intents(message, has_document=True)
        agents = {step["agent"] for step in plan["steps"]}
        self.assertEqual(agents, {"rag_agent", "tool_agent"})

    def test_reconcile_replaces_rag_only_weather_plan(self):
        steps = [
            {
                "id": "step_1",
                "agent": "rag_agent",
                "goal": "今天北京的天气怎么样？",
                "parallel_group": 0,
                "approval_required": False,
                "depends_on": [],
                "output_key": "document_answer",
            }
        ]
        reconciled, reason = reconcile_plan_steps(
            steps,
            "今天北京的天气怎么样？",
            document_id="doc-123",
        )
        self.assertEqual(len(reconciled), 1)
        self.assertEqual(reconciled[0]["agent"], "tool_agent")
        self.assertIn("tool_agent", reason)


if __name__ == "__main__":
    unittest.main()
