#!/usr/bin/env python3
"""Debug the tileset renderer: create an env from a .des, build shim agent, call renderer and print diagnostics."""
from pprint import pprint
DES = '/workspaces/NetPlay/scenarios/creativity/carry.des'

from netplay.nethack_utils.nle_wrapper import NethackGymnasiumWrapper
env = None
try:
    env = NethackGymnasiumWrapper(render_mode='human', des_file=DES, autopickup=False)
    obs, info = env.reset()
    print('env reset ok')
except Exception as e:
    print('failed to create env:', e)
    raise

# Build shim agent similar to test
class EnvAgentShim:
    def __init__(self, env):
        self.env = env
        try:
            obs, info = env.reset()
            self.last_observation = obs
        except Exception:
            try:
                self.last_observation = env.env.last_observation
            except Exception:
                self.last_observation = {}

        try:
            import numpy as _np
            g = _np.array(self.last_observation.get('glyphs'))
            rows, cols = g.shape
            class B: pass
            b = B()
            b.x = cols // 2
            b.y = rows // 2
            self.blstats = b
        except Exception:
            class B: pass
            b = B(); b.x = 0; b.y = 0; self.blstats = b
    def describe_current_state(self):
        return 'Game state summary.\n\nAgent Information:\n- HP: unknown\n- Location unknown'

agent = EnvAgentShim(env)

# Import renderer and call
from netplay.nethack_agent.skill_selection import render_tileset_map_cropped

print('Calling render_tileset_map_cropped...')
img, legend = render_tileset_map_cropped(agent, radius=8)
print('Result types:', type(img), type(legend))
if img is None:
    print('Renderer returned None, message:', legend)
else:
    try:
        import numpy as _np
        print('Image shape:', _np.array(img).shape)
    except Exception as e:
        print('Error inspecting image:', e)

print('Legend (first 200 chars):', (legend or '')[:200])

# Also attempt to introspect GlyphMapper and tiles
print('\nIntrospect GlyphMapper and tiles')
try:
    import numpy as _np
    from minihack.tiles.glyph_mapper import GlyphMapper
    from minihack.tiles import glyph2tile
    gm = GlyphMapper()
    tiles = gm.tiles
    print('Loaded GlyphMapper tiles type:', type(tiles))
    if isinstance(tiles, dict):
        k = next(iter(tiles.keys()))
        print('Sample tile key:', k, 'tile shape:', tiles[k].shape)
    else:
        print('Tiles array shape:', tiles.shape)
    print('glyph2tile length:', len(glyph2tile))
except Exception as e:
    print('Failed to inspect GlyphMapper:', e)

# Print a small sample of the cropped glyphs
import numpy as _np
g = _np.array(agent.last_observation['glyphs'])
print('glyphs shape:', g.shape)
center = (agent.blstats.y if hasattr(agent.blstats,'y') else g.shape[0]//2,
          agent.blstats.x if hasattr(agent.blstats,'x') else g.shape[1]//2)
print('center coords used:', center)
print('center glyph id:', g[center[0], center[1]])

print('\nDone')
