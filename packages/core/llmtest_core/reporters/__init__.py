"""
Reporters — Output test results in various formats.

Design: Each reporter takes a TestSuiteResult and produces output.
Reporters are stateless and side-effect-free (they return strings/dicts).
The caller decides where to write the output.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from llmtest_core.models import TestResult, TestStatus, TestSuiteResult


class BaseReporter(ABC):
    """Abstract base for all reporters."""

    @abstractmethod
    def render(self, suite: TestSuiteResult) -> str:
        """Render the test suite result as a string."""
        ...


class JSONReporter(BaseReporter):
    """
    JSON reporter for programmatic consumption.
    Perfect for CI/CD pipelines, dashboards, and trend tracking.
    """

    def __init__(self, indent: int = 2, include_raw: bool = False):
        self._indent = indent
        self._include_raw = include_raw

    def render(self, suite: TestSuiteResult) -> str:
        return json.dumps(self.to_dict(suite), indent=self._indent, default=str)

    def to_dict(self, suite: TestSuiteResult) -> dict[str, Any]:
        duration = (suite.finished_at - suite.started_at) if suite.finished_at else 0
        return {
            "suite_name": suite.suite_name,
            "summary": {
                "total": suite.total,
                "passed": suite.passed,
                "failed": suite.failed,
                "error": sum(1 for r in suite.results if r.status == TestStatus.ERROR),
                "skipped": sum(1 for r in suite.results if r.status == TestStatus.SKIPPED),
                "total_cost_usd": round(suite.total_cost_usd, 6),
                "duration_seconds": round(duration, 2),
                "ci_should_fail": suite.ci_should_fail,
            },
            "tests": [self._test_to_dict(r) for r in suite.results],
            "timestamp": datetime.fromtimestamp(suite.started_at).isoformat(),
        }

    def _test_to_dict(self, result: TestResult) -> dict[str, Any]:
        d: dict[str, Any] = {
            "test_id": result.test_id,
            "test_name": result.test_name,
            "status": result.status.value,
            "duration_ms": round(result.duration_ms, 1),
            "tags": result.tags,
        }

        if result.llm_output:
            d["llm_output"] = {
                "model": result.llm_output.model,
                "latency_ms": round(result.llm_output.latency_ms, 1),
                "input_tokens": result.llm_output.input_tokens,
                "output_tokens": result.llm_output.output_tokens,
                "cost_estimate_usd": round(result.llm_output.cost_estimate_usd, 6),
                "finish_reason": result.llm_output.finish_reason,
            }
            if self._include_raw:
                d["llm_output"]["content"] = result.llm_output.content

        if result.assertion_results:
            d["assertions"] = [
                {
                    "name": a.assertion_name,
                    "passed": a.passed,
                    "severity": a.severity.value,
                    "reason": a.reason,
                    "expected": a.expected,
                    "actual": a.actual,
                }
                for a in result.assertion_results
            ]

        if result.error:
            d["error"] = result.error

        return d


class JUnitReporter(BaseReporter):
    """
    JUnit XML reporter for CI systems (GitHub Actions, Jenkins, GitLab CI).

    Output format follows the JUnit XML schema so CI systems can parse it
    and display test results natively.
    """

    def render(self, suite: TestSuiteResult) -> str:
        duration = (suite.finished_at - suite.started_at) if suite.finished_at else 0

        testsuites = ET.Element("testsuites")
        testsuite = ET.SubElement(
            testsuites,
            "testsuite",
            {
                "name": suite.suite_name,
                "tests": str(suite.total),
                "failures": str(suite.failed),
                "errors": str(sum(1 for r in suite.results if r.status == TestStatus.ERROR)),
                "skipped": str(sum(1 for r in suite.results if r.status == TestStatus.SKIPPED)),
                "time": f"{duration:.3f}",
                "timestamp": datetime.fromtimestamp(suite.started_at).isoformat(),
            },
        )

        for result in suite.results:
            testcase = ET.SubElement(
                testsuite,
                "testcase",
                {
                    "name": result.test_name,
                    "classname": f"llmtest.{suite.suite_name}",
                    "time": f"{result.duration_ms / 1000:.3f}",
                },
            )

            if result.status == TestStatus.FAILED:
                failure_msgs = []
                for a in result.failed_assertions:
                    msg = f"[{a.severity.value.upper()}] {a.assertion_name}: {a.reason}"
                    if a.suggestions:
                        msg += " | Suggestions: " + "; ".join(a.suggestions)
                    failure_msgs.append(msg)
                failure = ET.SubElement(
                    testcase,
                    "failure",
                    {
                        "message": "; ".join(a.reason for a in result.failed_assertions),
                        "type": "AssertionFailure",
                    },
                )
                failure.text = "\n".join(failure_msgs)

            elif result.status == TestStatus.ERROR:
                error = ET.SubElement(
                    testcase,
                    "error",
                    {
                        "message": result.error or "Unknown error",
                        "type": "LLMError",
                    },
                )
                error.text = result.error

            elif result.status == TestStatus.SKIPPED:
                ET.SubElement(testcase, "skipped")

            # Add LLM metadata as properties
            if result.llm_output:
                props = ET.SubElement(testcase, "properties")
                ET.SubElement(
                    props,
                    "property",
                    {
                        "name": "llm.model",
                        "value": result.llm_output.model,
                    },
                )
                ET.SubElement(
                    props,
                    "property",
                    {
                        "name": "llm.latency_ms",
                        "value": f"{result.llm_output.latency_ms:.0f}",
                    },
                )
                ET.SubElement(
                    props,
                    "property",
                    {
                        "name": "llm.cost_usd",
                        "value": f"{result.llm_output.cost_estimate_usd:.6f}",
                    },
                )
                ET.SubElement(
                    props,
                    "property",
                    {
                        "name": "llm.input_tokens",
                        "value": str(result.llm_output.input_tokens),
                    },
                )
                ET.SubElement(
                    props,
                    "property",
                    {
                        "name": "llm.output_tokens",
                        "value": str(result.llm_output.output_tokens),
                    },
                )

        return ET.tostring(testsuites, encoding="unicode", xml_declaration=True)


class ConsoleReporter(BaseReporter):
    """
    Rich terminal output. Used by CLI and pytest plugin.
    """

    def __init__(self, verbose: bool = False, use_color: bool = True):
        self._verbose = verbose
        self._use_color = use_color

    def render(self, suite: TestSuiteResult) -> str:
        lines: list[str] = []
        duration = (suite.finished_at - suite.started_at) if suite.finished_at else 0

        lines.append("")
        lines.append(self._sep("llmtest summary"))
        lines.append(f"  Tests: {suite.passed} passed, {suite.failed} failed, {suite.total} total")
        lines.append(f"  Cost:  ${suite.total_cost_usd:.6f}")
        lines.append(f"  Time:  {duration:.1f}s")
        lines.append("")

        for result in suite.results:
            icon = self._status_icon(result.status)
            line = f"  {icon} {result.test_name}"
            if result.llm_output:
                latency = result.llm_output.latency_ms
                cost = result.llm_output.cost_estimate_usd
                line += f"  ({latency:.0f}ms, ${cost:.5f})"
            lines.append(line)

            if self._verbose and result.failed_assertions:
                for a in result.failed_assertions:
                    lines.append(
                        f"      [{a.severity.value.upper()}] {a.assertion_name}: {a.reason}"
                    )

        lines.append("")
        return "\n".join(lines)

    def _sep(self, title: str) -> str:
        total_width = 60
        padding = total_width - len(title) - 2
        left = padding // 2
        right = padding - left
        return f"{'─' * left} {title} {'─' * right}"

    def _status_icon(self, status: TestStatus) -> str:
        icons = {
            TestStatus.PASSED: "✓",
            TestStatus.FAILED: "✗",
            TestStatus.ERROR: "⚠",
            TestStatus.SKIPPED: "○",
        }
        return icons.get(status, "?")
