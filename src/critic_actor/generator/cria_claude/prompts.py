"""
Claude Agent SDK Client prompts for Critic-Actor
"""


def get_system_prompt_template(name: str) -> str:
    """
    Get a prompt template by name
    """
    if name == "human_tool":
        return PROMPT_TEMPLATE_HUMAN_TOOL
    else:
        raise ValueError(f"Sorry, '{name}' prompt template not implemented yet.")


PROMPT_TEMPLATE_HUMAN_TOOL = """
You are helping the user complete a task.

The user has instructions and a set of tools they can call. These are provided below.  

Your objective is to respond with {num_actions} different messages, each suggesting a different possible next tool to call.

Each message should include:
- "reasoning": a short description of the thought process for the suggested step
- "tool_call": the suggested tool call. For the user to understand, this must be formatted as the following text:
'''
<tool_call>
{{"name": "<tool_name>", "arguments": {{"<arg>": "<value>"}}}}
</tool_call>
'''

The user will then respond with their chosen step and the outcome that resulted.

# User Instructions:
'''
{system_prompt}
'''

# User Tools:
'''
{tools}
'''
""".strip()