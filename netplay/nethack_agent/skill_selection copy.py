import netplay.nethack_agent.skills as sk
from netplay.nethack_agent.agent import NetHackAgent, finish_task_skill
from netplay.core.skill_repository import SkillRepository, Skill

from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, BaseMessage

import json
import jsonschema
import os
from copy import deepcopy
from dataclasses import dataclass
from textwrap import dedent
from typing import Tuple, Dict, Any, Optional, List

skill_call_schema = {
    "type" : "object",
    "properties" : {
        "thoughts": {
            "type": "object",
            "properties": {
                "observations": {type: "string"},
                "reasoning": {type: "string"},
                "speak": {type: "string"}
            },
            "required": ["observations", "reasoning", "speak"],
            "additionalProperties": False
        },
        "skill": {
            "type": "object",
            "properties": {
                "name": {type: "string"},
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


def render_ascii_map_cropped(agent: NetHackAgent, radius: int) -> str:
    """Return an ASCII map cropped around the agent position with given radius.

    Strategy:
    - Read `agent.last_observation['tty_chars']` (array of rows of ints).
    - Find the first occurrence of the agent char '@' (ord('@') == 64) in the
      tty buffer. If not found, fall back to the center of the buffer.
    - Crop `radius` characters in each direction (rows and columns) around the
      found position, replacing null bytes with spaces so alignment is preserved.
    - Return the cropped block as a newline-joined string.

    This is robust to missing observations and will return a short placeholder
    if anything goes wrong.
    """
    try:
        tty = agent.last_observation["tty_chars"]
    except Exception:
        return "MAP HERE"

    try:
        nrows = len(tty)
        ncols = len(tty[0]) if nrows > 0 else 0

        agent_row = None
        agent_col = None
        at_code = ord("@")
        # Search for the '@' glyph in the tty buffer
        for r_index, row in enumerate(tty):
            for c_index, cell in enumerate(row):
                try:
                    if int(cell) == at_code:
                        agent_row, agent_col = r_index, c_index
                        break
                except Exception:
                    continue
            if agent_row is not None:
                break

        # Fallback to center if not found
        if agent_row is None:
            agent_row = nrows // 2
            agent_col = ncols // 2

        # Compute crop bounds
        r0 = max(0, agent_row - radius)
        r1 = min(nrows, agent_row + radius + 1)
        c0 = max(0, agent_col - radius)
        c1 = min(ncols, agent_col + radius + 1)

        lines = []
        for r in range(r0, r1):
            row = tty[r]
            # Keep alignment: replace zero bytes with space
            chars = [chr(int(c)) if int(c) != 0 else " " for c in row[c0:c1]]
            lines.append("".join(chars))

        # Trim empty rows/columns to reduce whitespace while keeping the agent in view
        # Convert to list of lists for easier trimming
        grid = [list(line) for line in lines]
        if len(grid) == 0:
            return "MAP HERE"

        nrows = len(grid)
        ncols = len(grid[0]) if nrows > 0 else 0

        # Find non-space bounding box
        min_r, max_r = nrows, -1
        min_c, max_c = ncols, -1
        for ri in range(nrows):
            for ci in range(ncols):
                if grid[ri][ci] != ' ':
                    if ri < min_r: min_r = ri
                    if ri > max_r: max_r = ri
                    if ci < min_c: min_c = ci
                    if ci > max_c: max_c = ci

        # If map is entirely blank, just return the centered grid with annotation
        if max_r == -1:
            cropped = [''.join(row) for row in grid]
        else:
            # Expand bounding box slightly to give context (one cell padding)
            min_r = max(0, min_r - 1)
            max_r = min(nrows - 1, max_r + 1)
            min_c = max(0, min_c - 1)
            max_c = min(ncols - 1, max_c + 1)
            cropped = [''.join(grid[ri][min_c:max_c+1]) for ri in range(min_r, max_r+1)]

        # Build a display with row/column mini-coordinates (relative to agent)
        try:
            # Determine the bounds in the original tty coordinates for the cropped region
            crop_r0 = r0 + (min_r if max_r != -1 else 0)
            crop_c0 = c0 + (min_c if max_c != -1 else 0)

            # Compute column headers as absolute map X coordinates using agent.blstats.x
            cols = len(cropped[0]) if len(cropped) > 0 else 0
            col_coords = []
            for j in range(cols):
                orig_c = crop_c0 + j
                # Convert tty column difference to map X by adding difference to agent.blstats.x
                abs_x = agent.blstats.x + (orig_c - agent_col)
                col_coords.append(abs_x)

            # Compute row headers as absolute map Y coordinates using agent.blstats.y
            rows = len(cropped)
            row_coords = []
            for i in range(rows):
                orig_r = crop_r0 + i
                abs_y = agent.blstats.y + (orig_r - agent_row)
                row_coords.append(abs_y)

            # Build display lines without axis headers: keep raw cropped rows
            display_lines = list(cropped)
            # Remove the agent marker from the displayed map to avoid duplication
            # (the agent position is reported separately in the prompt). Replace
            # '@' with '.' to show the underlying floor instead of the player glyph.
            display_lines = [ln.replace('@', '.') for ln in display_lines]

            # Build legend from characters present in the displayed map
            # (after removing the '@' marker) so the agent glyph isn't listed
            # in the legend when we replace it in the view.
            present = set(''.join(display_lines))
            # Authoritative legend mapping based on NetHack Guidebook (section 3)
            # Comprehensive legend mapping based on NetHack Guidebook (section 3)
            # Only entries for glyphs that appear in the cropped map will be shown.
            legend_map = {
                '@': 'you',
                '.': 'floor (room floor, ice, or doorway)',
                '#': 'corridor (or bars/tree/sink)',
                '-': 'horizontal wall',
                '|': 'vertical wall',
                '+': 'closed door',
                '<': 'stairs up',
                '>': 'stairs down',
                '$': 'pile of gold',
                '^': 'trap',
                ')': 'weapon',
                '[': 'armor',
                '%': 'edible (food)',
                '?': 'scroll',
                '/': 'wand',
                '=': 'ring',
                '!': 'potion',
                '(': 'tool / useful item (pick-axe, key, lamp, etc.)',
                '"': 'amulet or (web)',
                '*': 'gem or rock',
                '`': 'boulder or statue',
                '0': 'iron ball',
                '_': 'altar or iron chain',
                '{': 'fountain',
                '}': 'pool of water / lava / moat',
                '\\': 'throne',
                ',': 'object on floor (item)',
                ':': 'lizard (monster glyph)',
                ';': 'eel / sea monster (monster glyph)',
                '~': 'worm tail (part of long worm)',
                '\'': 'quote (rare symbol)',
                '#': 'corridor',
                'v': 'vortex or monster glyph',
                'V': 'vampire or monster glyph',
                '*': 'gem',
                '"': 'amulet or web',
                ' ': 'unseen (not observed / out of view)'
            }

            legend_lines = []
            for ch in sorted(present):
                # Letters and many other characters represent monsters/creatures
                if ch.isalpha():
                    desc = 'monster/creature'
                else:
                    desc = legend_map.get(ch, 'other')
                legend_lines.append(f"'{ch}': {desc}")

            # Return cropped map and legend (one entry per line). The agent
            # location annotation is handled by the prompt assembler to avoid
            # duplication with the agent information block.
            legend_block = "\n".join(legend_lines)
            return "\n".join(display_lines) + "\nLegend:\n" + legend_block
        except Exception:
            # If anything goes wrong building the fancy view, fall back to simple output
            try:
                annotation = f"Agent at ({agent.blstats.x}, {agent.blstats.y})"
            except Exception:
                annotation = "Agent position unknown"
            return "\n".join(cropped) + "\n" + annotation
    except Exception:
        return "MAP HERE"


def assemble_prompt(agent: NetHackAgent, skills: SkillRepository, task: str, map_radius: int = 10) -> str:
    """Assemble the final prompt including state, a map section, skills, and the task.

    This wraps `construct_prompt` but injects an additional 'Map' block between
    the state description and the skills list. Use this function where the
    full prompt (including map) should be sent to the LLM.
    """
    state_description = agent.describe_current_state()
    map_str = render_ascii_map_cropped(agent, map_radius)

    # Try to insert the Map block above the 'Agent Information:' section so
    # the map appears before the agent summary. If that marker is not present
    # fall back to inserting above 'Rooms:'; if neither marker exists, prepend
    # the map at the top.
    agent_marker = "\n\nAgent Information:"
    rooms_marker = "\n\nRooms:"
    insert_block = f"\n\nMap:\n{map_str}"
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

def render_tileset_map_cropped(agent: NetHackAgent, radius: int, tile_size: int = 32):
    """Construct a cropped map image (RGB) around the agent using the
    project's tileset and glyph-to-tile mapping. This uses the existing
    tileset (provided by minihack) to build the rendered map — it does not
    create or download a tileset file.

    Returns a tuple (image_array, legend_text). `image_array` is a HxWx3
    numpy array (uint8) suitable for saving or displaying. `legend_text` is
    a newline-separated string with one legend entry per line describing the
    glyphs present in the cropped region.

    The function imports GlyphMapper and `describe_glyph` lazily so that
    modules which only need ASCII rendering don't require the tiles
    dependency at module import time.
    """
    try:
        # Local imports to avoid hard dependency at module import time
        import numpy as _np
        from minihack.tiles.glyph_mapper import GlyphMapper
        from minihack.tiles import glyph2tile as _glyph2tile
        from netplay.nethack_agent.describe import describe_glyph
    except Exception:
        return None, "MAP HERE"

    try:
        glyphs = agent.last_observation["glyphs"]
    except Exception:
        return None, "MAP HERE"

    try:
        glyphs_arr = _np.array(glyphs)
        try:
            center_r = int(agent.blstats.y)
            center_c = int(agent.blstats.x)
        except Exception:
            center_r = glyphs_arr.shape[0] // 2
            center_c = glyphs_arr.shape[1] // 2

        r0 = max(0, center_r - radius)
        r1 = min(glyphs_arr.shape[0], center_r + radius + 1)
        c0 = max(0, center_c - radius)
        c1 = min(glyphs_arr.shape[1], center_c + radius + 1)

        cropped = glyphs_arr[r0:r1, c0:c1]
        if cropped.size == 0:
            return None, "MAP HERE"

        # Use GlyphMapper's tiles (loaded from tiles.pkl) and the glyph2tile mapping
        try:
            mapper = GlyphMapper()
            tiles = mapper.tiles  # expected shape: (n_tiles, th, tw, 3)
        except Exception:
            return None, "MAP HERE"

        if tiles is None or len(tiles) == 0:
            return None, "MAP HERE"

        rows, cols = cropped.shape

        # tiles may be a dict (index -> ndarray) or a numpy array
        if isinstance(tiles, dict):
            sample_tile = next(iter(tiles.values()))
            th, tw = int(sample_tile.shape[0]), int(sample_tile.shape[1])
            dtype = sample_tile.dtype
            fallback_tile = _np.zeros_like(sample_tile)

            def _get_tile(idx: int):
                return tiles.get(idx, fallback_tile)
        else:
            th = int(tiles.shape[1])
            tw = int(tiles.shape[2])
            dtype = tiles.dtype

            def _get_tile(idx: int):
                i = max(0, min(int(idx), len(tiles) - 1))
                return tiles[i]

        canvas = _np.zeros((rows * th, cols * tw, 3), dtype=dtype)

        g2t = _np.array(_glyph2tile)
        # Build image mosaic from cropped glyphs
        for ry in range(rows):
            for cx in range(cols):
                gid = int(cropped[ry, cx])
                if gid < 0:
                    tile_idx = 0
                elif gid >= len(g2t):
                    tile_idx = int(g2t[-1]) if len(g2t) > 0 else 0
                else:
                    tile_idx = int(g2t[gid])

                tile_img = _get_tile(tile_idx)
                y0 = ry * th
                x0 = cx * tw
                # Ensure tile_img shape matches expected tile size
                try:
                    canvas[y0:y0 + th, x0:x0 + tw] = tile_img
                except Exception:
                    # If shapes mismatch, fill with fallback
                    canvas[y0:y0 + th, x0:x0 + tw] = fallback_tile

        # Legend only for glyphs present in the cropped region (exclude agent glyph)
        unique_glyphs = sorted(_np.unique(cropped).tolist())
        legend_lines = []
        try:
            agent_glyph = int(glyphs_arr[center_r, center_c])
        except Exception:
            agent_glyph = None

        for gid in unique_glyphs:
            if agent_glyph is not None and int(gid) == int(agent_glyph):
                continue
            try:
                desc = describe_glyph(int(gid)) or "unknown"
            except Exception:
                desc = "unknown"
            legend_lines.append(f"{int(gid)}: {desc}")

        legend_text = "\n".join(legend_lines)
        return canvas, legend_text
    except Exception:
        return None, "MAP HERE"



