"""
System prompt and tool descriptions for the GDPval environment.
"""

SYSTEM_PROMPT_TEMPLATE = """\
You are a professional assistant completing a real-world work task. You have access to a Python execution environment (like Jupyter) where variables persist across calls.

## Task

{task_prompt}

## Working Directory

Your working directory is: {workdir}
All output files should be saved in this directory.

{reference_files_section}

## Available Python Packages

You have access to common data science and office document packages including:
- pandas, numpy, matplotlib, seaborn (data analysis and visualization)
- openpyxl, xlsxwriter (Excel files)
- python-pptx (PowerPoint files)
- pdfplumber, reportlab, fpdf (PDF files)
- json, csv, re, datetime, pathlib (standard library)

## Rules

- Use the `execute_python` tool to run Python code. Variables persist between calls.
- When you have produced all deliverable files, call the `submit` tool.
- Save all output files to the working directory.
- Do not give up. If something fails, debug and try again.
- You have a limited number of turns, so be efficient.
- Focus on producing high-quality deliverables that match the task requirements.\
"""

REFERENCE_FILES_SECTION = """\
## Reference Files

The following reference files are available in your working directory:
{file_list}

You can read them using pandas, openpyxl, or other appropriate libraries.\
"""

NO_REFERENCE_FILES_SECTION = """\
## Reference Files

No reference files were provided for this task. Generate all content from scratch.\
"""

EXECUTE_PYTHON_TOOL_DESC = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": (
            "Execute Python code in a persistent namespace. Variables, imports, "
            "and definitions persist across calls (like Jupyter notebook cells). "
            "Use this to process data, create files, and produce deliverables."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute.",
                }
            },
            "required": ["code"],
        },
    },
}

SUBMIT_TOOL_DESC = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": (
            "Submit your deliverables for grading. Call this when you have produced "
            "all required output files in the working directory. The grader will "
            "evaluate your files against the task rubric."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

TOOLS = [EXECUTE_PYTHON_TOOL_DESC, SUBMIT_TOOL_DESC]


def build_system_prompt(task_prompt: str, workdir: str, reference_files: list[str]) -> str:
    """Build the system prompt for a GDPval episode.

    Args:
        task_prompt: The task description.
        workdir: Path to the working directory.
        reference_files: List of reference file names available.

    Returns:
        Formatted system prompt string.
    """
    if reference_files:
        file_list = "\n".join(f"- {f}" for f in reference_files)
        ref_section = REFERENCE_FILES_SECTION.format(file_list=file_list)
    else:
        ref_section = NO_REFERENCE_FILES_SECTION

    return SYSTEM_PROMPT_TEMPLATE.format(
        task_prompt=task_prompt,
        workdir=workdir,
        reference_files_section=ref_section,
    )
