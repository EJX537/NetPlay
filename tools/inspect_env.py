#!/usr/bin/env python3
from pprint import pprint
from netplay.nethack_utils.nle_wrapper import NethackGymnasiumWrapper

DES = '/workspaces/NetPlay/scenarios/creativity/carry.des'

try:
    env = NethackGymnasiumWrapper(render_mode='human', des_file=DES, autopickup=False)
    obs, info = env.reset()
    print('reset done. obs type:', type(obs))
    try:
        print('obs keys:', list(obs.keys()))
    except Exception as e:
        print('cannot list obs keys:', e)

    # Check wrapped env last_observation
    try:
        wrapped = env.env
        print('wrapped env type:', type(wrapped))
        if hasattr(wrapped, 'last_observation'):
            print('wrapped.last_observation present')
            lo = wrapped.last_observation
            try:
                print('last_observation keys (if dict):', list(lo.keys()) if isinstance(lo, dict) else 'not a dict')
            except Exception:
                pass
        else:
            print('wrapped has no last_observation')
    except Exception as e:
        print('error inspecting wrapped env:', e)

    # Print glyphs shape if available in obs
    try:
        if 'glyphs' in obs:
            g = obs['glyphs']
            import numpy as _np
            g_arr = _np.array(g)
            print('glyphs shape:', g_arr.shape)
            print('glyphs sample at center:', g_arr[g_arr.shape[0]//2, g_arr.shape[1]//2])
        else:
            print('glyphs not in obs')
    except Exception as e:
        print('error reading glyphs from obs:', e)

    try:
        if 'blstats' in obs:
            print('blstats len:', len(obs['blstats']))
            print('blstats sample:', obs['blstats'])
        else:
            print('blstats not in obs')
    except Exception as e:
        print('error reading blstats from obs:', e)

except Exception as e:
    print('failed to create/reset env:', e)
    raise

print('\nDone')
