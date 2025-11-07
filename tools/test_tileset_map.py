#!/usr/bin/env python3
"""Small test utility to produce a tileset-based map PNG and the assembled prompt for a
synthetic state.

This script creates a fake agent observation (a small glyphs grid using valid
glyph ids from minihack), calls `render_tileset_map_cropped`, saves the resulting
PNG and legend into `runs/interactive/`, and writes the assembled prompt to a
text file so you can inspect exactly what would be sent to the LLM.
"""
import os
import argparse

def _make_agent_from_env(env):
    # Build a tiny shim object that satisfies the renderer's expected interface
    class EnvAgentShim:
        def __init__(self, env):
            self.env = env
            # Reset the env to obtain the observation dict (wrapped env may store a tuple)
            try:
                obs, info = env.reset()
                self.last_observation = obs
            except Exception:
                # Fallback: try to read wrapped last_observation and convert to dict if possible
                try:
                    self.last_observation = env.env.last_observation
                except Exception:
                    self.last_observation = {}

            # Compute a safe blstats-like object. Prefer the environment-provided
            # blstats (if present) so we can extract HP and an accurate location.
            class B: pass
            b = B()
            b.x = 0
            b.y = 0
            b.hitpoints = None
            b.max_hitpoints = None
            try:
                bl = None
                if isinstance(self.last_observation, dict):
                    bl = self.last_observation.get('blstats')
                if bl is not None:
                    try:
                        from netplay.nethack_agent.tracking import make_blstats
                        bl_named = make_blstats(bl)
                        b.x = getattr(bl_named, 'x', b.x)
                        b.y = getattr(bl_named, 'y', b.y)
                        b.hitpoints = getattr(bl_named, 'hitpoints', None)
                        b.max_hitpoints = getattr(bl_named, 'max_hitpoints', None)
                    except Exception:
                        # If make_blstats is not available or fails, fall back
                        # to center-based coordinates below
                        bl = None

                if bl is None:
                    import numpy as _np
                    g = _np.array(self.last_observation.get('glyphs'))
                    rows, cols = g.shape
                    b.x = cols // 2
                    b.y = rows // 2
            except Exception:
                b.x = 0; b.y = 0

            self.blstats = b

        def describe_current_state(self):
            # Provide HP and Location when available (fall back to unknown)
            try:
                hp = self.blstats.hitpoints
                max_hp = self.blstats.max_hitpoints
            except Exception:
                hp = None
                max_hp = None

            try:
                loc_x = int(self.blstats.x)
                loc_y = int(self.blstats.y)
                loc = f"({loc_x},{loc_y})"
            except Exception:
                loc = "unknown"

            if hp is not None and max_hp is not None:
                hp_str = f"{hp}/{max_hp}"
            elif hp is not None:
                hp_str = str(hp)
            else:
                hp_str = "unknown"

            return (
                "Game state summary.\n\n"
                "Agent Information:\n"
                f"- HP: {hp_str}\n"
                f"- Location: {loc}"
            )

    return EnvAgentShim(env)


def main():
    out_dir = os.path.join("runs", "interactive")
    os.makedirs(out_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--des', dest='des', default=None, help='Optional .des file to load via MiniHack')
    parser.add_argument('--task', dest='task', default=None, help='Optional task text to include in the prompt')
    args = parser.parse_args()

    agent = None
    if args.des:
        try:
            # Create a real env from the des-file to obtain a realistic observation
            from netplay.nethack_utils.nle_wrapper import NethackGymnasiumWrapper
            # Use 'human' render mode here to avoid requiring the monospace font
            # file that may not be present in this environment. We only need the
            # observation data from the env, not the ASCII rendering.
            env = NethackGymnasiumWrapper(render_mode='human', des_file=args.des, autopickup=False)
            print(env)
            obs, info = env.reset()
            # Some wrappers populate last_observation on the wrapped env object
            # Wrap this in a small agent-like shim
            agent = _make_agent_from_env(env)
        except Exception as e:
            print("Failed to create env from des-file, falling back to synthetic agent:", e)
            return
    else:
        print("No .des file provided, using synthetic agent observation.")
        return

    print(agent.describe_current_state())

    # Import the functions we just added to the repo
    from netplay.nethack_agent.skill_selection import render_tileset_map_cropped, assemble_prompt, CHOOSE_SKILL_PROMPT
    from netplay.core.skill_repository import SkillRepository

    img, legend = render_tileset_map_cropped(agent, radius=8)

    if img is None:
        print("Tileset render failed or missing dependency. Message:", legend)
        return

    # Save image (this is the map image constructed using the tileset)
    try:
        from PIL import Image
        Image.fromarray(img).save(os.path.join(out_dir, "test_tiles_map.png"))
        print(f"Saved tileset-based map PNG to {os.path.join(out_dir, 'test_tiles_map.png')}")
    except Exception as e:
        print("Failed to save image:", e)

    # Save legend
    with open(os.path.join(out_dir, "test_tiles_legend.txt"), "w") as f:
        f.write(legend or "")
    print(f"Saved legend to {os.path.join(out_dir, 'test_tiles_legend.txt')}")

    # Build prompt and save.
    # For the tileset-based test we don't need to include the ASCII map inline in
    # the prompt (images will be handled by VLMs in a separate test). If the
    # tileset renderer produced an image, use `construct_prompt` which does not
    # inject the ASCII map; otherwise fall back to `assemble_prompt`.
    from netplay.nethack_agent.skill_selection import construct_prompt
    # Build a SkillRepository containing all skills declared in the skills module
    import netplay.nethack_agent.skills as skill_module
    skill_list = [getattr(skill_module, name) for name in dir(skill_module)
                  if hasattr(getattr(skill_module, name), 'skill')]
    repo = SkillRepository(skill_list)
    # If this test used a des-file env, synthesize a fuller state description
    # from the observation so the prompt contains everything normally present
    # at a level's start (blstats, inventory, messages, visible glyphs).
    # Decide what to use as the 'task' section in the prompt. If the user
    # provided a --task string prefer that, otherwise fall back to the
    # interactive CHOOSE_SKILL_PROMPT which asks the model to pick a skill.
    task_text = args.task if getattr(args, 'task', None) else CHOOSE_SKILL_PROMPT

    if img is not None and args.des is not None:
        try:
            import numpy as _np
            from netplay.nethack_agent.describe import describe_glyph
            # Use the helper to pretty-print blstats when available
            try:
                from netplay.nethack_agent.tracking import make_blstats
            except Exception:
                make_blstats = None

            obs = agent.last_observation
            # blstats may be an array-like
            bl = obs.get('blstats') if isinstance(obs, dict) else None
            if bl is not None and make_blstats is not None:
                try:
                    bl_named = make_blstats(bl)
                    bl_lines = [f"- {name}: {getattr(bl_named, name)}" for name in bl_named._fields]
                    bl_str = '\n'.join(bl_lines)
                except Exception:
                    bl_str = str(list(bl))
            else:
                bl_str = str(list(bl)) if bl is not None else 'unknown'

            inv = obs.get('inv_strs') or []
            inv_lines = '\n'.join([f'- {s}' for s in inv]) if len(inv) > 0 else 'None'

            last_msg = ''
            try:
                m = obs.get('message')
                if isinstance(m, (list, tuple)) and len(m) > 0:
                    last_msg = str(m[0])
                else:
                    last_msg = str(m)
            except Exception:
                last_msg = ''

            # Visible glyphs: crop around center like the renderer
            try:
                g = _np.array(obs.get('glyphs'))
                rows, cols = g.shape
                cx = cols // 2
                cy = rows // 2
                radius = 8
                r0 = max(0, cy - radius)
                r1 = min(rows, cy + radius + 1)
                c0 = max(0, cx - radius)
                c1 = min(cols, cx + radius + 1)
                cropped = g[r0:r1, c0:c1]
                unique = sorted(_np.unique(cropped).tolist())
                glyph_lines = []
                for gid in unique:
                    try:
                        desc = describe_glyph(int(gid))
                    except Exception:
                        desc = 'unknown'
                    glyph_lines.append(f"- {int(gid)}: {desc}")
                glyph_block = '\n'.join(glyph_lines) if len(glyph_lines) > 0 else 'None'
            except Exception:
                glyph_block = 'None'

            # Determine a sensible location string: prefer bl_named (from blstats),
            # otherwise fall back to the agent shim's blstats values if present.
            loc_str = 'unknown'
            try:
                if bl is not None and make_blstats is not None:
                    loc_str = f"({int(bl_named.x)},{int(bl_named.y)})"
                else:
                    # Try EnvAgentShim's computed blstats
                    try:
                        loc_str = f"({int(agent.blstats.x)},{int(agent.blstats.y)})"
                    except Exception:
                        loc_str = 'unknown'
            except Exception:
                loc_str = 'unknown'

            state_description = '\n\n'.join([
                'Game state summary.',
                'Agent Information:\n' + (bl_str if bl_str else '- blstats: unknown') + f"\n- Location: {loc_str}",
                'Inventory:\n' + inv_lines,
                'Last message:\n' + (last_msg or 'None'),
                'Visible glyphs:\n' + glyph_block
            ])
            prompt = construct_prompt(state_description, repo, task_text)
        except Exception:
            prompt = construct_prompt(agent.describe_current_state(), repo, task_text)
    else:
        # Fallback: use agent.describe_current_state for non-des or failed cases
        prompt = assemble_prompt(agent, repo, task_text, map_radius=8)
    with open(os.path.join(out_dir, "test_prompt.txt"), "w") as f:
        f.write(prompt)
    print(f"Saved prompt to {os.path.join(out_dir, 'test_prompt.txt')}")

    # Write a simple manifest JSON useful for VLM pipelines referencing the
    # produced artifacts. Paths are relative to the workspace.
    try:
        manifest = {
            "png": os.path.join(out_dir, "test_tiles_map.png"),
            "legend": os.path.join(out_dir, "test_tiles_legend.txt"),
            "prompt": os.path.join(out_dir, "test_prompt.txt"),
            "task": task_text if task_text else ""
        }
        import json
        with open(os.path.join(out_dir, "test_tiles_manifest.json"), "w") as mf:
            json.dump(manifest, mf, indent=2)
        print(f"Saved manifest to {os.path.join(out_dir, 'test_tiles_manifest.json')}")
    except Exception as e:
        print("Failed to write manifest:", e)


if __name__ == "__main__":
    main()
