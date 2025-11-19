# PNG Map Mode Implementation

## Overview
PNG map mode is now fully implemented and working! This feature enables vision-capable LLMs to see tileset-rendered maps of the game environment.

## What Was Implemented

### 1. Image Generation (`map_renderer.py`)
- Uses existing `render_tileset_map_cropped()` function
- Renders NetHack glyphs using the tileset from minihack
- Returns numpy array (HxWx3) with RGB image data
- Generates legend of visible objects

### 2. Image Encoding (`skill_selection.py`)
- Converts numpy array to PIL Image
- Encodes as PNG in memory (no file I/O)
- Converts to base64 data URL format
- Stores on agent as `_pending_image_data`

### 3. Vision Model Support (`llm_wrapper.py`)
- Enhanced `predict_messages()` to handle image content
- Checks for `image_data` in `message.additional_kwargs`
- Formats messages for vision models using LiteLLM's standard format:
  ```python
  {
    "role": "user",
    "content": [
      {"type": "text", "text": "..."},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }
  ```

### 4. Prompt Integration (`skill_selection.py`)
- Adds map reference to text prompt
- Includes legend of visible objects
- Properly integrates with existing prompt structure

## Supported Vision Models

The implementation works with any LiteLLM-supported vision model:
- `gemini/gemini-2.0-flash-exp` ✓
- `gemini/gemini-1.5-pro` ✓
- `gpt-4-vision-preview` ✓
- `gpt-4o` ✓
- `claude-3-opus` ✓
- `claude-3-sonnet` ✓

## Usage

```python
from netplay import create_llm_agent, MapMode
from netplay.llm_wrapper import LiteLLMWrapper

# Create vision-capable LLM
llm = LiteLLMWrapper(model='gemini/gemini-2.0-flash-exp')

# Create agent with PNG mode
agent = create_llm_agent(
    env=env,
    llm=llm,
    map_mode='png',      # or MapMode.PNG
    map_radius=10        # crop radius
)
```

## Testing

Run the test script to verify:
```bash
python test_png_mode.py
```

This will:
- Generate a tileset map image
- Encode it as base64
- Create a prompt with image reference
- Save decoded image to `runs/test_png/generated_map.png`

## Example Output

The vision model receives:
1. **Text prompt** with game state, inventory, skills, etc.
2. **PNG image** of the cropped tileset map (e.g., 336x320 pixels)
3. **Legend** listing all visible objects in the map

Example prompt snippet:
```
Map Image:
[A tileset-rendered map image is provided showing the area around you]

Visible Objects (from map):
2306: iron wand
2359: dark area
2360: vertical wall
2361: horizontal wall
...
5913: newt statue
```

## Benefits Over ASCII Mode

1. **Visual clarity**: Tileset graphics are clearer than ASCII
2. **Spatial reasoning**: Vision models can better understand layout
3. **Object recognition**: Tiles are more distinct than text symbols
4. **No token overhead**: Image data doesn't consume text tokens

## Files Modified

1. `netplay/llm_wrapper.py` - Added image content support
2. `netplay/nethack_agent/skill_selection.py` - PNG mode implementation
3. `example_map_modes.py` - Updated documentation
4. `test_png_mode.py` - New test script
5. `example_png_vision.py` - New usage example

## Notes

- Image size is typically ~2-3KB base64 encoded
- Map is cropped around agent position (configurable radius)
- Falls back gracefully if tileset rendering fails
- No external files required - all in memory
