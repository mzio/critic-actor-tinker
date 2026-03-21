"""
Prompt templates for the Chat Claude generator.

Two prompt contexts:
1. Policy LLM: gets an instruction appended to the task system prompt so it
   generates a "chat message" (tip / reflection) instead of calling tools directly.
2. Claude: receives the original task context plus the policy's chat message,
   and produces a single concrete tool-call action.

Template variants:
- "chat": policy provides a free-form tip / reflection
- "handoff": policy writes a structured handoff note (situation, key findings,
  recommended action) as if briefing a colleague who will execute
"""


def get_policy_chat_instruction(tools_str: str, name: str = "chat") -> str:
    """
    Return the instruction block appended to the policy LLM's system prompt.
    Tells the policy to produce a chat message instead of calling tools directly.
    """
    if name == "chat":
        return POLICY_CHAT_INSTRUCTION.format(tools=tools_str)
    if name == "handoff":
        return POLICY_HANDOFF_INSTRUCTION.format(tools=tools_str)
    raise ValueError(f"Sorry, '{name}' policy chat instruction template not implemented yet.")


def get_claude_system_prompt_template(name: str) -> str:
    if name == "chat":
        return CLAUDE_SYSTEM_PROMPT_TEMPLATE_CHAT
    if name == "handoff":
        return CLAUDE_SYSTEM_PROMPT_TEMPLATE_HANDOFF
    raise ValueError(f"Sorry, '{name}' prompt template not implemented yet.")


def get_claude_chat_prefix(chat_text: str, name: str = "chat") -> str:
    """
    Return a prefix string that is prepended to the per-turn user prompt
    sent to Claude. This injects the policy's chat message each turn while
    keeping the system prompt (and therefore Claude's conversation state) fixed.
    """
    if name == "chat":
        return CLAUDE_CHAT_PREFIX_CHAT.format(policy_chat_message=chat_text)
    if name == "handoff":
        return CLAUDE_CHAT_PREFIX_HANDOFF.format(policy_chat_message=chat_text)
    raise ValueError(f"Sorry, '{name}' prefix template not implemented yet.")


# ===================================================================
# "chat" variant — free-form tip / reflection
# ===================================================================


POLICY_CHAT_SYSTEM_PROMPT_TEMPLATE = """
You are a guidance model. You are helping the user complete a task.

The user has instructions and a set of tools they can call. These are provided below.

# User Instructions
'''
{system_prompt}

{instruction_prompt}
'''

# User Tools
'''
{tools}
'''

# Requirements
Instead of calling tools directly, analyze the current situation and provide a short chat message
— a tip or reflection — about what should be done next.

First think about how to provide the best tip or reflection. This should guide 
the user in their decision making on what to do next, and may include:
- A brief analysis of the current state and what information is available
- The most important factors or new information learned
- Your reasoning for why this is the best tip or reflection

Then respond in the following format:
'''
<final_response>
{{final-response}}
</final_response>
'''
""".strip()

# POLICY_CHAT_INSTRUCTION = """

# # Your Role
# You are a guidance model. Instead of calling tools directly, analyze the current
# situation and provide a short chat message — a tip or reflection — about what
# should be done next.

# Your message should include:
# - A brief analysis of the current state and what information is available
# - A specific recommendation for which tool to call next and with what arguments
# - Your reasoning for why this is the best next step

# Do NOT wrap your response in tool call tags. Just provide your analysis and
# recommendation as plain text.
# """.strip()
POLICY_CHAT_INSTRUCTION = """

# Your Role
You are a guidance model. Instead of calling tools directly, analyze the current
situation and provide a short chat message — a tip or reflection — about what
should be done next.

First think about how to provide the best tip or reflection. This should guide 
the user in their decision making on what to do next, and may include:
- A brief analysis of the current state and what information is available
- The most important factors or new information learned
- Your reasoning for why this is the best tip or reflection

Then include your final tip or reflection in the following format:
<final_response>
{final-response}
</final_response>
""".strip()


CLAUDE_SYSTEM_PROMPT_TEMPLATE_CHAT = """
You are helping the user complete a task.

The user has instructions and a set of tools they can call. These are provided below.

# User Instructions
'''
{system_prompt}
'''

# User Tools
'''
{tools}
'''

Respond with a single tool call describing what the user should do next. Include:
- "reasoning": a short description of your thought process
- "tool_call": the tool call formatted as:
'''
<tool_call>
{{"name": "<tool_name>", "arguments": {{"<arg>": "<value>"}}}}
</tool_call>
'''

Each turn, the user may also provide a tip or reflection on what to do next.
Use the tip to inform your action, but apply your own judgment — it may be imperfect.
""".strip()


# Per-turn prefix injected before the conversation prompt
CLAUDE_CHAT_PREFIX_CHAT = """
[Helper Tip]
{policy_chat_message}
""".strip()


# ===================================================================
# "handoff" variant — structured briefing note
# ===================================================================

POLICY_HANDOFF_INSTRUCTION = """

# Your Role
You are handing off this task to a colleague who will execute the next step.
Write a short, structured handoff note so they can act immediately.

Your note MUST contain these sections:
1. **Situation** — one or two sentences summarizing where things stand right now
   (what the task is, what has already been done, what data is on the table).
2. **Key Findings** — bullet the most important facts, numbers, or observations
   from the conversation so far that are relevant to the next step.
3. **Tips** — any tips for the next step. Do not include any explicit action instructions.
   Be specific enough that your colleague can act without re-reading the full conversation.

Think and reason about how to provide the best handoff note. 

Then include your final handoff note in the following format:
<final_response>
{final-response}
</final_response>
""".strip()


# CLAUDE_SYSTEM_PROMPT_TEMPLATE_HANDOFF = """
# You are helping the user complete a task.

# The user has instructions and a set of tools they can call. These are provided below.

# # User Instructions
# '''
# {system_prompt}
# '''

# # User Tools
# '''
# {tools}
# '''

# Respond with a single tool call. Include:
# - "reasoning": a short description of your thought process
# - "tool_call": the tool call formatted as:
# '''
# <tool_call>
# {{"name": "<tool_name>", "arguments": {{"<arg>": "<value>"}}}}
# </tool_call>
# '''

# Each turn, the user will also provide a handoff note summarizing the situation and
# providing any tips or recommendations. Use the briefing to inform your action, but apply
# your own judgment — the recommendation may need adjustment.
# """.strip()
CLAUDE_SYSTEM_PROMPT_TEMPLATE_HANDOFF = """
You are helping the user complete a task.

The user has instructions and a set of tools they can call. These are provided below.

# User Instructions
'''
{system_prompt}
'''

# User Tools
'''
{tools}
'''

To help the user, you must respond in the following format:
'''
[a short description of your thought process]
<tool_call>
{{"name": "<tool_name>", "arguments": {{"<arg>": "<value>"}}}}
</tool_call>
'''

## Example:
'''
I need to gather information about All National Insurance's underwriting guidelines for workers compensation, particularly for manufacturing businesses.
My approach will be to:
1) Retrieve the company's underwriting guidelines to check appetite for workers comp in manufacturing
2) Explore the database tables to understand what risk data is available
3) Compare FlowMeter Precision Devices against the stated guidelines.

I'll start by getting the underwriting guidelines and listing available tables.
<tool_call>
{{"name": "get_underwriting_guidelines", "arguments": {{}}}}
</tool_call>
'''

Each turn, the user will also provide a handoff note summarizing the situation and
providing any tips or recommendations. Use the briefing to inform your action, but apply
your own judgment — the recommendation may need adjustment.
""".strip()


# Per-turn prefix injected before the conversation prompt
CLAUDE_CHAT_PREFIX_HANDOFF = """
[Handoff Note]
{policy_chat_message}
""".strip()
