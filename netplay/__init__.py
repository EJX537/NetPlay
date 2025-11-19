import netplay.nethack_agent.skills as skills
import netplay.nethack_agent.descriptors as descriptors
import netplay.nethack_agent.skill_selection as skill_selection
from netplay.nethack_agent.agent import NetHackAgent
from netplay.nethack_agent.skill_selection import MapMode
from netplay.core.skill_repository import SkillRepository
from netplay.core.descriptor import TitleValueDescriptor

# Export MapMode for external use
__all__ = ['create_llm_agent', 'MapMode']

def create_llm_agent(env, llm, memory_tokens, log_folder, render=False,
                     censor_nethack_context=False, enable_finish_task_skill=True,
                     update_hidden_objects=False, map_mode="none", map_radius=10):
    """Create an LLM-powered NetHack agent.

    Args:
        env: The NetHack environment
        llm: The language model wrapper
        memory_tokens: Maximum number of tokens for message history
        log_folder: Folder to save logs
        render: Whether to render the environment visually
        censor_nethack_context: Whether to censor "NetHack" references
        enable_finish_task_skill: Whether to enable the finish_task skill
        update_hidden_objects: Whether to update hidden object tracking
        map_mode: Map rendering mode - "none" (default), "ascii", or "png"
        map_radius: Radius of the map to render (default 10)

    Returns:
        Configured NetHackAgent instance
    """
    # Convert string map_mode to MapMode enum
    if isinstance(map_mode, str):
        map_mode_map = {
            "none": MapMode.NONE,
            "ascii": MapMode.ASCII,
            "png": MapMode.PNG
        }
        map_mode_enum = map_mode_map.get(map_mode.lower(), MapMode.NONE)
    else:
        map_mode_enum = map_mode

    skill_repo = SkillRepository([
        *skills.ALL_COMMAND_SKILLS,
        skills.set_avoid_monster_flag,
        skills.melee_attack,
        skills.explore_level,
        skills.move_to,
        skills.go_to,
        skills.press_key,
        skills.type_text,
    ])
    state_descriptor = TitleValueDescriptor({
        "Context": descriptors.GeneralContextDescriptor() if censor_nethack_context else descriptors.NetHackContextDescriptor(),
        "Agent Information": descriptors.AgentInformationDescriptor(),
        "Rooms": descriptors.RoomsObjectFeatureDescriptor(),
        "Close Monsters": descriptors.CloseMonsterDescriptor(),
        "Distant Monsters": descriptors.DistantMonsterDescriptor(),
        #"Current Room": descriptors.CurrentRoomDescriptor(),
        #"Other Rooms": descriptors.OtherRoomsDescriptor(),
        "Exploration Status": descriptors.ExplorationStatusDescriptor(),
        "Inventory": descriptors.InventoryDescriptor(),
        "Stats": descriptors.StatsDescriptor(),
        "Task": descriptors.TaskDescriptor()
    })
    skill_selector = skill_selection.SimpleSkillSelector(
        llm=llm,
        skills=skill_repo,
        use_popup_prompt=True,
        map_mode=map_mode_enum,
        map_radius=map_radius
    )
    agent = NetHackAgent(
        env=env,
        state_descriptor=state_descriptor,
        skill_selector=skill_selector,
        llm=llm,
        skills=skill_repo,
        max_memory_tokens=memory_tokens,
        log_folder=log_folder,
        render=render,
        censor_nethack_messages=censor_nethack_context,
        enable_finish_task_skill=enable_finish_task_skill,
        update_hidden_objects=update_hidden_objects
    )
    return agent
