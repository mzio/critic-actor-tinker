"""
LLM rubric grader for GDPval.

Grades deliverable files against a rubric using an LLM judge. Supports batch
mode (single call with all criteria) and per-criterion mode (one call per
criterion).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File preview generation
# ---------------------------------------------------------------------------


def preview_file(path: Path, max_chars: int = 8000) -> str:
    """Generate a text preview of a file for LLM grading.

    Supports: xlsx/xls (openpyxl), csv/txt/py/json (text), pdf (pdfplumber),
    pptx (python-pptx), binary fallback.
    """
    suffix = path.suffix.lower()

    try:
        if suffix in (".xlsx", ".xls"):
            return _preview_spreadsheet(path, max_chars)
        elif suffix == ".pdf":
            return _preview_pdf(path, max_chars)
        elif suffix == ".pptx":
            return _preview_pptx(path, max_chars)
        elif suffix in (".csv", ".txt", ".py", ".json", ".md", ".html", ".xml", ".yaml", ".yml"):
            return _preview_text(path, max_chars)
        else:
            size = path.stat().st_size
            return f"[Binary file: {path.name}, {size} bytes]"
    except Exception as e:
        return f"[Error previewing {path.name}: {e}]"


_PREVIEW_ROWS = 21  # header + 20 data rows


def _preview_spreadsheet(path: Path, max_chars: int) -> str:
    """Preview an Excel file using openpyxl."""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    parts = [f"Excel file: {path.name}"]
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        parts.append(f"\n--- Sheet: {sheet_name} ---")
        for _row_idx, row in enumerate(ws.iter_rows(max_row=_PREVIEW_ROWS, values_only=True)):
            parts.append("\t".join(str(c) if c is not None else "" for c in row))
        if ws.max_row is not None and ws.max_row > _PREVIEW_ROWS:
            parts.append(f"... ({ws.max_row - _PREVIEW_ROWS} more rows)")
    wb.close()
    text = "\n".join(parts)
    return text[:max_chars] if len(text) > max_chars else text


def _preview_pdf(path: Path, max_chars: int) -> str:
    """Preview a PDF file using pdfplumber."""
    import pdfplumber

    parts = [f"PDF file: {path.name}"]
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages[:10]):
            text = page.extract_text() or ""
            if text.strip():
                parts.append(f"\n--- Page {i + 1} ---")
                parts.append(text)
    text = "\n".join(parts)
    return text[:max_chars] if len(text) > max_chars else text


def _preview_pptx(path: Path, max_chars: int) -> str:
    """Preview a PPTX file using python-pptx."""
    from pptx import Presentation

    prs = Presentation(path)
    parts = [f"PowerPoint file: {path.name}"]
    for i, slide in enumerate(prs.slides):
        slide_text = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                slide_text.append(shape.text)
        if slide_text:
            parts.append(f"\n--- Slide {i + 1} ---")
            parts.extend(slide_text)
    text = "\n".join(parts)
    return text[:max_chars] if len(text) > max_chars else text


def _preview_text(path: Path, max_chars: int) -> str:
    """Preview a text file."""
    text = path.read_text(errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... (truncated {len(text) - max_chars} chars)"
    return text


# ---------------------------------------------------------------------------
# Grading prompt templates
# ---------------------------------------------------------------------------

BATCH_GRADING_PROMPT = """\
You are grading a task submission. The task was:

{task_prompt}

The rubric has the following criteria:

{rubric_text}

Here are the deliverable files produced:

{file_previews}

For each criterion, determine if it is satisfied based on the deliverable files.
Respond with a JSON array where each element has:
- "criterion_index": the 0-based index of the criterion
- "satisfied": true or false
- "reasoning": brief explanation

Respond with ONLY the JSON array, no other text.
"""

PER_CRITERION_PROMPT = """\
You are grading a task submission on a single criterion.

Task: {task_prompt}

Criterion ({points} points): {criterion_name}
Description: {criterion_description}

Deliverable files:

{file_previews}

Is this criterion satisfied? Respond with a JSON object:
{{"satisfied": true/false, "reasoning": "brief explanation"}}

Respond with ONLY the JSON object, no other text.
"""


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------


class GDPvalGrader:
    """LLM rubric grader for GDPval tasks.

    Args:
        grader_model_config: Config dict for loading the grader LLM via ``load_llm``.
        batch_criteria: If True, grade all criteria in one LLM call.
            If False, make one call per criterion.
        max_new_tokens: Max tokens for grader response.
    """

    def __init__(
        self,
        grader_model_config: dict[str, Any] | None = None,
        batch_criteria: bool = True,
        max_new_tokens: int = 4096,
    ) -> None:
        from ...llm_handlers import load_llm

        if grader_model_config is None:
            grader_model_config = {
                "name": "openai",
                "model_config": {"model": "gpt-4o"},
            }
        self.grader_model = load_llm(**grader_model_config)
        self.batch_criteria = batch_criteria
        self.max_new_tokens = max_new_tokens

    def grade(
        self,
        task_prompt: str,
        rubric_json: list[dict],
        deliverable_paths: list[Path],
        total_points: int,
    ) -> tuple[float, dict[str, Any]]:
        """Grade deliverables against the rubric.

        Args:
            task_prompt: The original task description.
            rubric_json: List of rubric criterion dicts.
            deliverable_paths: Paths to files produced by the model.
            total_points: Maximum possible points.

        Returns:
            ``(reward, info)`` where reward is in [0, 1].
        """
        if not deliverable_paths:
            return 0.0, {"earned_points": 0, "total_points": total_points, "criteria_results": []}

        # Build file previews
        file_previews = []
        for path in deliverable_paths:
            preview = preview_file(path)
            file_previews.append(f"=== {path.name} ===\n{preview}")
        file_preview_text = "\n\n".join(file_previews)

        if self.batch_criteria:
            return self._grade_batch(task_prompt, rubric_json, file_preview_text, total_points)
        else:
            return self._grade_per_criterion(
                task_prompt, rubric_json, file_preview_text, total_points
            )

    def _grade_batch(
        self,
        task_prompt: str,
        rubric_json: list[dict],
        file_preview_text: str,
        total_points: int,
    ) -> tuple[float, dict[str, Any]]:
        """Grade all criteria in a single LLM call."""
        rubric_lines = []
        for i, c in enumerate(rubric_json):
            name = c.get("criterion", c.get("name", f"Criterion {i}"))
            points = c.get("points", 1)
            desc = c.get("description", "")
            rubric_lines.append(f"[{i}] ({points}pt) {name}: {desc}")
        rubric_text = "\n".join(rubric_lines)

        prompt = BATCH_GRADING_PROMPT.format(
            task_prompt=task_prompt,
            rubric_text=rubric_text,
            file_previews=file_preview_text,
        )

        response_text = self._call_llm(prompt)
        criteria_results = self._parse_batch_response(response_text, rubric_json)

        earned = sum(
            rubric_json[r["criterion_index"]].get("points", 1)
            for r in criteria_results
            if r.get("satisfied")
        )
        reward = earned / total_points if total_points > 0 else 0.0

        return reward, {
            "earned_points": earned,
            "total_points": total_points,
            "criteria_results": criteria_results,
            "grader_response": response_text,
        }

    def _grade_per_criterion(
        self,
        task_prompt: str,
        rubric_json: list[dict],
        file_preview_text: str,
        total_points: int,
    ) -> tuple[float, dict[str, Any]]:
        """Grade each criterion individually."""
        criteria_results = []
        earned = 0

        for i, criterion in enumerate(rubric_json):
            name = criterion.get("criterion", criterion.get("name", f"Criterion {i}"))
            points = criterion.get("points", 1)
            desc = criterion.get("description", "")

            prompt = PER_CRITERION_PROMPT.format(
                task_prompt=task_prompt,
                points=points,
                criterion_name=name,
                criterion_description=desc,
                file_previews=file_preview_text,
            )

            response_text = self._call_llm(prompt)
            satisfied = self._parse_criterion_response(response_text)

            if satisfied:
                earned += points

            criteria_results.append(
                {
                    "criterion_index": i,
                    "satisfied": satisfied,
                    "reasoning": response_text,
                }
            )

        reward = earned / total_points if total_points > 0 else 0.0
        return reward, {
            "earned_points": earned,
            "total_points": total_points,
            "criteria_results": criteria_results,
        }

    def _call_llm(self, prompt: str) -> str:
        """Call the grader LLM and return the response text."""
        messages = [{"role": "user", "content": prompt}]
        response = self.grader_model.sample(
            system_prompt="You are an expert grader. Respond only with valid JSON.",
            messages=messages,
            tools=None,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=1,
        )[0]
        actions = self.grader_model.get_actions(response)
        return actions[-1].text if actions else ""

    def _parse_batch_response(
        self, response_text: str, rubric_json: list[dict]
    ) -> list[dict[str, Any]]:
        """Parse batch grading response into criterion results."""
        try:
            # Try to extract JSON array from response
            text = response_text.strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            results = json.loads(text)
            if isinstance(results, list):
                return results
        except (json.JSONDecodeError, ValueError):
            logger.warning("gdpval: failed to parse batch grading response")

        # Fallback: treat all as not satisfied, flag the parse failure
        return [
            {
                "criterion_index": i,
                "satisfied": False,
                "reasoning": "Parse error",
                "parse_failed": True,
            }
            for i in range(len(rubric_json))
        ]

    def _parse_criterion_response(self, response_text: str) -> bool:
        """Parse a single criterion grading response."""
        try:
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            result = json.loads(text)
            return bool(result.get("satisfied", False))
        except (json.JSONDecodeError, ValueError):
            # Fallback: stricter regex to avoid false positives like "not satisfied"
            match = re.search(r'"satisfied"\s*:\s*true\b', response_text, re.IGNORECASE)
            return bool(match)
