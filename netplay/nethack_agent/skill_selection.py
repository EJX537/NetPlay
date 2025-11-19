import netplay.nethack_agent.skills as sk
from netplay.nethack_agent.agent import NetHackAgent, finish_task_skill
from netplay.core.skill_repository import SkillRepository, Skill
from netplay.nethack_agent.map_renderer import render_ascii_map_cropped, render_tileset_map_cropped

from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage, BaseMessage

import json
import jsonschema
from copy import deepcopy
from dataclasses import dataclass
from textwrap import dedent
from typing import Tuple, Dict, Any, Optional, List
from enum import Enum


class MapMode(Enum):
    """Map rendering mode for the agent's visual context."""
    NONE = "none"      # No map rendering (original implementation)
    ASCII = "ascii"    # ASCII text-based map with semantic legend
    PNG = "png"        # PNG tileset rendering with image embedding for vision models

skill_call_schema = {
    "type": "object",
    "properties": {
        "thoughts": {
            "type": "object",
            "properties": {
                "observations": {"type": "string"},
                "reasoning": {"type": "string"},
                "speak": {"type": "string"}
            },
            "required": ["observations", "reasoning", "speak"],
            "additionalProperties": False
        },
        "skill": {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }
    },
    "additionalProperties": False,
    "required": ["thoughts", "skill"]
}

CHOOSE_SKILL_PROMPT = dedent("""
Choose an skill from the given list of skills.
Output your response in the following JSON format:
{
    "thoughts": {
        "observations": "<Relevant observations from your last action. Pay close attention to what you set out to do and compare that to the games current state.>",
        "reasoning": "<Plan ahead.>",
        "speak": "<Summary of thoughts, to say to user>"
    }
    "skill": {
        "name": "<The name of the skill>",
        "<param1_name>": "<The value for this parameter>",
        "<param2_name>": "<The value for this parameter>",
    }
}
""".strip())

POPUP_CHOOSE_SKILL_PROMPT = dedent("""
Resolve the popup by pressing keys.
If you want to close the popup abort it using ESC or confirm your choices using enter or space.
Output your response in the following JSON format:
{
    "thoughts": {
        "observations": "<Relevant observations from your last action. Pay close attention to what you set out to do and compare that to the games current state.>",
        "reasoning": "<Plan ahead.>",
        "speak": "<Summary of thoughts, to say to user>"
    }
    "skill": {
        "name": "<The name of the skill>",
        "<param1_name>": "<The value for this parameter>",
        "<param2_name>": "<The value for this parameter>",
    }
}
""".strip())

FIX_JSON_PROMPT = PromptTemplate(template=dedent("""
You were tasked to choose a skill from the given list of skills.
Your output:
{wrong_json}

Error message:
{error_message}

Fix the error and output your response in the following JSON format:
{{
    "thoughts": {{
        "observations": "<Relevant observations from your last action. Pay close attention to what you set out to do and compare that to the games current state.>",
        "reasoning": "<Plan ahead.>",
        "speak": "<Summary of thoughts, to say to user.>"
    }}
    "skill": {{
        "name": "<The name of the skill>",
        "<param1_name>": "<The value for this parameter>",
        "<param2_name>": "<The value for this parameter>",
    }}
}}
""".strip()), input_variables=["wrong_json", "error_message"])

CHOOSE_SKILL_LOG_FILE = "choose_skill_prompt.json"

@dataclass
class Thoughts:
    observations: str
    reasoning: str
    speak: str

@dataclass
class SkillSelection:
    thoughts: Thoughts
    skill: Skill
    skill_kwargs: Dict[str, Any]


def parse_json(json_str: str, skill_repo: SkillRepository) -> Tuple[Optional[Exception], Optional[SkillSelection]]:
    try:
        # Strip markdown code blocks if present (common with some LLMs like Gemini)
        json_str = json_str.strip()
        if json_str.startswith('```'):
            # Find the first newline after the opening ```
            first_newline = json_str.find('\n')
            if first_newline != -1:
                # Find the closing ```
                last_backticks = json_str.rfind('```')
                if last_backticks > first_newline:
                    json_str = json_str[first_newline + 1:last_backticks].strip()

        # Strip any text before the first { (LLM sometimes adds explanation before JSON)
        first_brace = json_str.find('{')
        if first_brace > 0:
            json_str = json_str[first_brace:]
        elif first_brace == -1:
            # No JSON found at all
            return "No JSON object found in response", None

        # Extract just the first complete JSON object to handle cases where LLM
        # returns multiple JSON objects or extra text after valid JSON
        brace_count = 0
        in_string = False
        escape_next = False
        end_pos = -1

        for i, char in enumerate(json_str):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
            elif not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break

        if end_pos > 0:
            json_str = json_str[:end_pos]

        json_dict = json.loads(json_str)
        jsonschema.validate(instance=json_dict, schema=skill_call_schema)
    except json.JSONDecodeError as e:
        return e.msg, None
    except jsonschema.ValidationError as e:
        return e.message, None

    # Verify the skills parameters
    skill_name = json_dict["skill"]["name"]
    kwargs = {name : value for name, value in json_dict["skill"].items() if name != "name"}
    try:
        skill = skill_repo.get_skill(skill_name)
        skill.verify_kwargs(kwargs)
    except ValueError as e:
        return str(e), None

    thoughts = Thoughts(**json_dict["thoughts"])
    return None, SkillSelection(thoughts=thoughts, skill=skill, skill_kwargs=kwargs)


def construct_prompt(state_description: str, skills: SkillRepository, task: str) -> str:
    return "\n\n".join([
        state_description,
        f"Skills:\n{skills.get_skills_description()}",
        task
    ])


def assemble_prompt_with_map(agent: NetHackAgent, skills: SkillRepository, task: str,
                              map_mode: MapMode, map_radius: int = 10) -> str:
    """Assemble the final prompt including state, optional map, skills, and task.

    Args:
        agent: The NetHack agent
        skills: Repository of available skills
        task: The task prompt to append
        map_mode: The map rendering mode (NONE, ASCII, or PNG)
        map_radius: Radius of the map to render (default 10)

    Returns:
        The assembled prompt string
    """
    state_description = agent.describe_current_state()

    # If no map mode, use original implementation
    if map_mode == MapMode.NONE:
        return construct_prompt(state_description, skills, task)

    # Render the appropriate map type
    if map_mode == MapMode.ASCII:
        map_str = render_ascii_map_cropped(agent, map_radius)
        insert_block = f"\n\nMap:\n{map_str}"
    elif map_mode == MapMode.PNG:
        # PNG mode: render tileset map and prepare for image embedding
        # Note: Image will be attached separately via message.additional_kwargs
        # The prompt text should mention the map image is provided
        from netplay.nethack_agent.map_renderer import render_tileset_map_cropped
        import base64
        from io import BytesIO

        img_array, legend = render_tileset_map_cropped(agent, map_radius)

        if img_array is not None:
            try:
                from PIL import Image
                import numpy as np

                # Convert numpy array to PIL Image
                img = Image.fromarray(np.uint8(img_array))

                # Encode to PNG and then base64
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                # Store the base64 image data in a way that can be accessed later
                # We'll attach it to the message via additional_kwargs
                agent._pending_image_data = f"data:image/png;base64,{img_base64}"

                # Add map reference and legend to the prompt
                map_reference = f"\n\nMap Image:\n[A tileset-rendered map image is provided showing the area around you]\n\nVisible Objects (from map):\n{legend}"

                # Insert the map reference before "Agent Information:" or "Rooms:" section
                agent_marker = "\n\nAgent Information:"
                rooms_marker = "\n\nRooms:"

                if agent_marker in state_description:
                    parts = state_description.split(agent_marker, 1)
                    modified_description = parts[0] + map_reference + agent_marker + parts[1]
                elif rooms_marker in state_description:
                    parts = state_description.split(rooms_marker, 1)
                    modified_description = parts[0] + map_reference + rooms_marker + parts[1]
                else:
                    modified_description = state_description + map_reference

                return construct_prompt(modified_description, skills, task)
            except Exception as e:
                # If image rendering fails, fall back to no map
                import traceback
                traceback.print_exc()
                return construct_prompt(state_description, skills, task)
        else:
            # Tileset rendering not available, fall back to no map
            return construct_prompt(state_description, skills, task)
    else:
        # Unknown mode, skip map
        return construct_prompt(state_description, skills, task)

    # Insert the map block before "Agent Information:" or "Rooms:" section
    agent_marker = "\n\nAgent Information:"
    rooms_marker = "\n\nRooms:"

    if agent_marker in state_description:
        idx = state_description.find(agent_marker)
        state_with_map = state_description[:idx] + insert_block + state_description[idx:]
    elif rooms_marker in state_description:
        idx = state_description.find(rooms_marker)
        state_with_map = state_description[:idx] + insert_block + state_description[idx:]
    else:
        # Fallback: place Map at the very top for visibility
        state_with_map = insert_block + "\n\n" + state_description

    return "\n\n".join([
        state_with_map,
        f"Skills:\n{skills.get_skills_description()}",
        task
    ])


class SimpleSkillSelector:
    def __init__(self,
        llm,
        skills: SkillRepository,
        use_popup_prompt: bool=False,
        map_mode: MapMode = MapMode.NONE,
        map_radius: int = 10
    ):
        self.llm = llm
        self.skills = skills
        self.use_popup_prompt = use_popup_prompt
        self.map_mode = map_mode
        self.map_radius = map_radius

    def choose_skill(self, agent: NetHackAgent) -> SkillSelection:
        if agent.waiting_for_popup() and self.use_popup_prompt:
            skills = [sk.press_key, sk.type_text]
            prompt = POPUP_CHOOSE_SKILL_PROMPT
        else:
            skills = agent.skills.skills.values()
            prompt = CHOOSE_SKILL_PROMPT

        if agent.enable_finish_task_skill:
            skills = [*skills, finish_task_skill]

        return self._internal_choose_skill(agent, SkillRepository(skills), prompt)

    def _internal_choose_skill(self, agent: NetHackAgent, skills: SkillRepository, prompt: str) -> SkillSelection:
        # Assemble prompt with or without map based on map_mode
        task_prompt = assemble_prompt_with_map(agent, skills, prompt, self.map_mode, self.map_radius)

        # Check if PNG mode generated an image to attach
        image_data = getattr(agent, '_pending_image_data', None)

        # Create the final message with optional image data
        if image_data:
            final_message = SystemMessage(
                content=task_prompt,
                additional_kwargs={'image_data': image_data}
            )
            # Clear the pending image data
            agent._pending_image_data = None
        else:
            final_message = SystemMessage(content=task_prompt)

        messages = [
            *agent.message_history.get_messages(),
            final_message
        ]        # Censoring
        if agent.censor_nethack_messages:
            messages = deepcopy(messages)
            for m in messages:
                m.content = m.content.replace("NetHack", "CENSORED")

        # Call and parse
        json_str = self.llm.predict_messages(messages).content
        agent.logger.log_json(
            data = {
                "prompt": messages[-1].content,
                "response": json_str,
                "context": [{m.type: m.content} for m in messages[:-1]]
            },
            file_name=CHOOSE_SKILL_LOG_FILE
        )

        error_message, skill_call = parse_json(json_str, skills)
        if error_message is None:
            return skill_call

        raise Exception(f"Unable to parse the JSON provided by the LLM. Error message: '{error_message}'.")

