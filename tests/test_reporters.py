"""Unit tests for reporters."""

import json
import xml.etree.ElementTree as ET

import pytest
from llmtest_core.models import (
    AssertionResult,
    LLMInput,
    LLMOutput,
    Severity,
    TestResult,
    TestStatus,
    TestSuiteResult,
)
from llmtest_core.reporters import ConsoleReporter, JSONReporter, JUnitReporter


@pytest.fixture
def sample_suite():
    return TestSuiteResult(
        suite_name="test_suite",
        results=[
            TestResult(
                test_id="t1",
                test_name="test_pass",
                status=TestStatus.PASSED,
                llm_input=LLMInput.simple("Hi"),
                llm_output=LLMOutput(
                    content="Hello",
                    model="gpt-4o",
                    latency_ms=150,
                    input_tokens=10,
                    output_tokens=5,
                ),
                assertion_results=[
                    AssertionResult(assertion_name="contains", passed=True, severity=Severity.HIGH),
                ],
                duration_ms=200,
            ),
            TestResult(
                test_id="t2",
                test_name="test_fail",
                status=TestStatus.FAILED,
                llm_input=LLMInput.simple("Hi"),
                llm_output=LLMOutput(
                    content="Wrong",
                    model="gpt-4o",
                    latency_ms=300,
                    input_tokens=10,
                    output_tokens=5,
                ),
                assertion_results=[
                    AssertionResult(
                        assertion_name="contains",
                        passed=False,
                        severity=Severity.HIGH,
                        reason="Expected 'Paris' not found",
                    ),
                ],
                duration_ms=350,
            ),
        ],
        started_at=1000.0,
        finished_at=1001.0,
    )


class TestJSONReporter:
    def test_renders_valid_json(self, sample_suite):
        reporter = JSONReporter()
        output = reporter.render(sample_suite)
        data = json.loads(output)
        assert data["suite_name"] == "test_suite"

    def test_summary(self, sample_suite):
        reporter = JSONReporter()
        data = reporter.to_dict(sample_suite)
        assert data["summary"]["total"] == 2
        assert data["summary"]["passed"] == 1
        assert data["summary"]["failed"] == 1

    def test_test_details(self, sample_suite):
        reporter = JSONReporter()
        data = reporter.to_dict(sample_suite)
        tests = data["tests"]
        assert len(tests) == 2
        assert tests[0]["status"] == "passed"
        assert tests[1]["status"] == "failed"

    def test_includes_assertions(self, sample_suite):
        reporter = JSONReporter()
        data = reporter.to_dict(sample_suite)
        assert "assertions" in data["tests"][1]
        assert data["tests"][1]["assertions"][0]["passed"] is False

    def test_include_raw(self, sample_suite):
        reporter = JSONReporter(include_raw=True)
        data = reporter.to_dict(sample_suite)
        assert "content" in data["tests"][0]["llm_output"]


class TestJUnitReporter:
    def test_renders_valid_xml(self, sample_suite):
        reporter = JUnitReporter()
        output = reporter.render(sample_suite)
        root = ET.fromstring(output)
        assert root.tag == "testsuites"

    def test_testsuite_attributes(self, sample_suite):
        reporter = JUnitReporter()
        output = reporter.render(sample_suite)
        root = ET.fromstring(output)
        suite = root.find("testsuite")
        assert suite.get("tests") == "2"
        assert suite.get("failures") == "1"

    def test_failure_details(self, sample_suite):
        reporter = JUnitReporter()
        output = reporter.render(sample_suite)
        root = ET.fromstring(output)
        failures = root.findall(".//failure")
        assert len(failures) == 1

    def test_properties(self, sample_suite):
        reporter = JUnitReporter()
        output = reporter.render(sample_suite)
        root = ET.fromstring(output)
        props = root.findall(".//property")
        assert len(props) > 0


class TestConsoleReporter:
    def test_renders_string(self, sample_suite):
        reporter = ConsoleReporter()
        output = reporter.render(sample_suite)
        assert "llmtest summary" in output
        assert "1 passed" in output
        assert "1 failed" in output

    def test_verbose_mode(self, sample_suite):
        reporter = ConsoleReporter(verbose=True)
        output = reporter.render(sample_suite)
        assert "contains" in output
