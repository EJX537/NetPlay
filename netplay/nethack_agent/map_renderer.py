"""Map rendering utilities for NetHack agent.

This module provides functions to render ASCII and tileset-based maps
cropped around the agent's position, with semantic legends based on
glyph descriptions.
"""

from typing import Tuple, Optional
import os


def render_ascii_map_cropped(agent, radius: int) -> str:
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

        # NetHack terminal layout (typical 24 rows):
        # Rows 0-1: Message area (game messages)
        # Row 2: Blank separator
        # Rows 3-21: Map area
        # Rows 22-23: Status lines
        # We only want the map area for the legend (rows 3-21)
        message_rows = 3  # Skip top 3 rows (message area)
        status_rows = 2   # Skip bottom 2 rows (status lines)
        map_start = message_rows
        map_end = max(message_rows, nrows - status_rows)

        agent_row = None
        agent_col = None
        at_code = ord("@")
        # Search for the '@' glyph in the tty buffer (only in map area)
        for r_index in range(map_start, map_end):
            row = tty[r_index]
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
            agent_row = (map_start + map_end) // 2
            agent_col = ncols // 2

        # Compute crop bounds (limited to map area, excluding messages and status)
        r0 = max(map_start, agent_row - radius)
        r1 = min(map_end, agent_row + radius + 1)  # Don't go past map area
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

            # Keep the @ symbol visible - don't replace it
            # The agent position is also reported in the Agent Information section,
            # but seeing it on the map provides important spatial context

            # Build legend from characters present in the displayed map
            present = set(''.join(display_lines))
            # Authoritative legend mapping based on NetHack Guidebook (section 3)
            # Comprehensive legend mapping based on NetHack Guidebook (section 3)
            # Only entries for glyphs that appear in the cropped map will be shown.
            # Based on NetHack Guidebook section 3.3 (official documentation)
            legend_map = {
                '@': 'you (or another human)',
                '.': 'floor of a room, ice, or doorless doorway',
                '#': 'corridor, or iron bars, or tree, or sink, or drawbridge',
                '-': 'wall of a room, or open door',
                '|': 'wall of a room, or open door, or grave',
                '+': 'closed door, or spellbook',
                '<': 'stairs up (to previous level)',
                '>': 'stairs down (to next level)',
                '$': 'pile of gold',
                '^': 'trap (detected)',
                ')': 'weapon',
                '[': 'suit or piece of armor',
                '%': 'something edible (not necessarily healthy)',
                '?': 'scroll',
                '/': 'wand',
                '=': 'ring',
                '!': 'potion',
                '(': 'useful item (pick-axe, key, lamp, etc.)',
                '"': 'amulet or spider web',
                '*': 'gem or rock (possibly valuable, possibly worthless)',
                '`': 'boulder or statue',
                '0': 'iron ball',
                '_': 'altar, or iron chain',
                '{': 'fountain',
                '}': 'pool of water or moat or pool of lava',
                '\\': 'opulent throne',
                ',': 'item on floor',
                'I': 'last known location of invisible/unseen monster'
            }

            # Enhance legend with semantic information from glyphs array
            # This helps disambiguate symbols that could have multiple meanings
            # (e.g., '`' could be boulder OR statue, letters could be any monster)
            try:
                from netplay.nethack_agent.describe import describe_glyph

                # Get the glyphs array - it's (21, 79) corresponding to map area only
                # (tty_chars rows 3-23, which is rows map_start to map_start+21)
                glyphs = agent.last_observation["glyphs"]

                # Build a mapping of char -> set of glyph descriptions
                # For the visible cropped area, collect what each character actually represents
                # Note: For the player (@), glyphs shows terrain underneath, not the player glyph
                char_to_descriptions = {}
                for i in range(rows):
                    orig_r = crop_r0 + i
                    for j in range(cols):
                        orig_c = crop_c0 + j
                        # Get the character from the cropped map
                        ch = cropped[i][j]
                        if ch not in (' ', '\n', '\t'):
                            # Special case: @ represents the player, not the terrain underneath
                            if ch == '@':
                                # Skip @ - it will use the legend_map entry
                                continue

                            # Convert tty row to glyphs row
                            # glyphs array starts at tty row 1 (not row 3/map_start)
                            # This is because glyphs includes message area rows as part of the map
                            glyph_r = orig_r - 1
                            glyph_c = orig_c

                            # Bounds check
                            if 0 <= glyph_r < len(glyphs) and 0 <= glyph_c < len(glyphs[0]):
                                glyph_id = int(glyphs[glyph_r][glyph_c])
                                desc = describe_glyph(glyph_id)
                                if desc:
                                    if ch not in char_to_descriptions:
                                        char_to_descriptions[ch] = set()
                                    char_to_descriptions[ch].add(desc)
            except Exception as e:
                # If we can't get glyph descriptions, fall back to basic legend
                char_to_descriptions = {}

            # Only show legend entries for symbols actually present in the map
            legend_lines = []

            # Check if there are any unmapped letters or numbers in the present set
            # According to NetHack docs: "Letters and certain other symbols represent
            # the various inhabitants of the Mazes of Menace"
            has_unmapped_chars = any(ch not in legend_map for ch in present if ch not in (' ', '\n', '\t'))

            for ch in sorted(present):
                if ch in legend_map:
                    # Show specific description for mapped symbols
                    desc = legend_map[ch]
                    # If we have semantic glyph information, add it
                    if ch in char_to_descriptions and len(char_to_descriptions[ch]) <= 3:
                        # Only show glyph details if there are 3 or fewer unique descriptions
                        # (to avoid overwhelming the legend with many monster types)
                        glyph_descs = sorted(char_to_descriptions[ch])
                        glyph_info = ", ".join(glyph_descs)
                        legend_lines.append(f"'{ch}': {desc} → {glyph_info}")
                    else:
                        legend_lines.append(f"'{ch}': {desc}")
                elif ch in char_to_descriptions and ch not in (' ', '\n', '\t'):
                    # For unmapped symbols, if we have glyph description, show it directly
                    glyph_descs = sorted(char_to_descriptions[ch])
                    if len(glyph_descs) <= 3:
                        glyph_info = ", ".join(glyph_descs)
                        legend_lines.append(f"'{ch}': {glyph_info}")

            # Add summary entry for letters and other monster symbols (only for those WITHOUT glyph descriptions)
            if has_unmapped_chars:
                # Collect specific monster types if available from glyph descriptions
                # But only for chars that weren't already listed above
                already_listed = set(ch for ch in present if ch in legend_map or ch in char_to_descriptions)
                unmapped_without_glyph = [ch for ch in present if ch not in already_listed and ch not in (' ', '\n', '\t')]

                monster_descs = set()
                for ch in unmapped_without_glyph:
                    if ch in char_to_descriptions:
                        monster_descs.update(char_to_descriptions[ch])

                if unmapped_without_glyph:
                    if monster_descs and len(monster_descs) <= 5:
                        # Show specific monsters if there are 5 or fewer types
                        monster_list = ", ".join(sorted(monster_descs))
                        legend_lines.append(f"letters (a-z, A-Z) and other symbols: {monster_list}")
                    else:
                        # Otherwise use generic description
                        legend_lines.append(f"letters (a-z, A-Z) and other symbols: various inhabitants of the Mazes of Menace (monsters)")

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


def render_tileset_map_cropped(agent, radius: int, tile_size: int = 32) -> Tuple[Optional[any], str]:
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
        try:
            from netplay.nethack_agent.describe import describe_glyph
        except Exception:
            # best-effort fallback
            def describe_glyph(g):
                return str(g)
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
            tiles = mapper.tiles  # expected shape: (n_tiles, th, tw, 3) or dict
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
                    # If shapes mismatch, fill with fallback if available
                    try:
                        canvas[y0:y0 + th, x0:x0 + tw] = _np.zeros((th, tw, 3), dtype=dtype)
                    except Exception:
                        pass

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
