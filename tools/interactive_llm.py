import os
from typing import List
from langchain.schema import BaseMessage, AIMessage
import json
import re

from langchain.chat_models import ChatOpenAI


class InteractiveLLM:
    """Wraps a real LLM (ChatOpenAI) to print the full messages passed and
    require user confirmation before issuing the real network call.

    If no real LLM is available or the user types 'manual' when prompted, the
    wrapper will accept a manual response string typed by the user and return
    it as an AIMessage.
    """
    def __init__(self, model_name: str = "gpt-4-1106-preview", temperature: float = 0.0):
        self.inner = None
        if ChatOpenAI is not None:
            try:
                self.inner = ChatOpenAI(model=model_name, temperature=temperature)
            except Exception:
                # Could not initialize (missing API key, etc.) — we'll fall back to manual mode
                self.inner = None

    def get_num_tokens(self, content: str) -> int:
        if content is None:
            return 0
        # If inner provides token counting, use a simple heuristic fallback
        return max(1, len(content) // 4)

    def predict_messages(self, messages: List[BaseMessage]):
        # Print the entire set of messages to console for inspection
        print('\n' + '='*80)
        print('LLM will be called with the following message list (full content):')
        for i, m in enumerate(messages):
            mt = type(m).__name__
            print(f"\n--- Message {i} ({mt}) ---")
            print(m.content)
        print('\n' + '='*80)

        while True:
            ans = input("Press Enter to call the real LLM, or type 'manual' to enter a manual response: ")
            if ans.strip().lower() == 'manual' or self.inner is None:
                print("Enter the response JSON/text to return to the agent (single line). Example JSON for skill selection:")
                print('{"thoughts": {"observations": "...", "reasoning": "...", "speak": "..."}, "skill": {"name": "press_key", "key": " "}}')
                manual = input('Response> ')
                return AIMessage(content=manual)

            # Try calling the inner LLM. If it errors (for example due to an invalid API key),
            # allow the user to input a new OPENAI_API_KEY and retry, or fall back to manual.
            try:
                return self.inner.predict_messages(messages)
            except Exception as e:
                print(f"LLM call failed with error: {e}")
                choice = input("Type 'retry' to try again, 'key' to enter a new OPENAI_API_KEY, or 'manual' to enter a manual response: ")
                if choice.strip().lower() == 'manual':
                    print("Enter the response JSON/text to return to the agent (single line). Example JSON for skill selection:")
                    manual = input('Response> ')
                    return AIMessage(content=manual)
                if choice.strip().lower() == 'key':
                    new_key = input('Enter new OPENAI_API_KEY value (will be set in the environment for this process): ')
                    if new_key.strip() != '':
                        os.environ['OPENAI_API_KEY'] = new_key.strip()
                        # Attempt to reinitialize the inner ChatOpenAI
                        try:
                            if ChatOpenAI is not None:
                                self.inner = ChatOpenAI(model=self.inner.model, temperature=self.inner.temperature)
                                print('Reinitialized LLM with new key, will retry.')
                                continue
                        except Exception as e2:
                            print(f'Failed to reinitialize LLM: {e2}')
                            # fallthrough to prompt choice again
                # otherwise loop and ask again (retry or new key)


class LoggingLLM:
    """Wrapper that logs exact prompts and responses to files and stdout.

    It delegates to an inner LLM which may implement predict_messages or invoke.
    """
    def __init__(self, inner_llm, log_folder: str, pause_before_call: bool = False):
        self.inner = inner_llm
        self.log_folder = log_folder
        os.makedirs(self.log_folder, exist_ok=True)
        self._count = 0
        self.pause_before_call = pause_before_call

    def get_num_tokens(self, content: str) -> int:
        if hasattr(self.inner, 'get_num_tokens'):
            try:
                return self.inner.get_num_tokens(content)
            except Exception:
                pass
        return max(1, len(content) // 4)

    def _serialize_messages(self, messages):
        parts = []
        for m in messages:
            mtype = type(m).__name__
            parts.append(f"--- {mtype} ---\n{m.content}\n")
        return "\n".join(parts)

    def predict_messages(self, messages):
        # Log prompt
        self._count += 1
        prompt_text = self._serialize_messages(messages)
        fname_base = os.path.join(self.log_folder, f"llm_call_{self._count}")
        try:
            with open(fname_base + '.prompt.txt', 'w') as f:
                f.write(prompt_text)
        except Exception:
            pass
        print('\n' + '='*40)
        print('LLM PROMPT (exact):')
        print(prompt_text)
        print('='*40 + '\n')

        if self.pause_before_call:
            input('Press Enter to call the LLM (or Ctrl-C to abort)')

    # Delegate to inner LLM (prefer predict_messages, fallback to invoke)
        try:
            if hasattr(self.inner, 'predict_messages'):
                resp = self.inner.predict_messages(messages)
            else:
                # many modern LLMs expose invoke; accept list of messages
                resp = self.inner.invoke(messages)
        except Exception as e:
            print(f'LLM call failed: {e}')
            raise

        # Serialize response exactly as returned
        resp_content = getattr(resp, 'content', None)
        resp_text = resp_content if resp_content is not None else str(resp)
        try:
            with open(fname_base + '.response_raw.txt', 'w') as f:
                f.write(resp_text)
        except Exception:
            pass

        # Attempt to extract a JSON object from the response (greedy) so the
        # downstream agent which expects JSON can parse it. If extraction
        # succeeds, return an AIMessage whose content is the cleaned JSON string.
        json_match = re.search(r"\{.*\}", resp_text, re.DOTALL)
        if json_match:
            candidate = json_match.group(0)
            try:
                parsed = json.loads(candidate)
                cleaned = json.dumps(parsed)
                try:
                    with open(fname_base + '.response.json', 'w') as f:
                        f.write(cleaned)
                except Exception:
                    pass

                print('\n' + '-'*40)
                print('LLM RESPONSE (exact):')
                print(resp_text)
                print('-'*40 + '\n')
                print('LLM RESPONSE (extracted JSON):')
                print(cleaned)
                print('-'*40 + '\n')

                # Return an AIMessage with only the JSON content so existing
                # parsing code can consume it.
                return AIMessage(content=cleaned)
            except Exception:
                # If JSON extraction fails, fall back to returning raw
                pass

        try:
            with open(fname_base + '.response.txt', 'w') as f:
                f.write(resp_text)
        except Exception:
            pass

        print('\n' + '-'*40)
        print('LLM RESPONSE (exact):')
        print(resp_text)
        print('-'*40 + '\n')

        return resp



def run_interactive(des_file: str = None, model: str = 'gpt-4-1106-preview', log_folder: str = None, headless: bool = False, step_pause: bool = False):
    """Start an interactive agent session using the InteractiveLLM wrapper.

    This mirrors the previous `tools/run_interactive_llm.py` but is consolidated here.
    """
    from netplay import create_llm_agent
    from netplay.nethack_utils.nle_wrapper import NethackGymnasiumWrapper

    if log_folder is None:
        log_folder = os.path.join(os.getcwd(), 'runs', 'interactive')
    os.makedirs(log_folder, exist_ok=True)

    llm = InteractiveLLM(model_name=model)
    # If headless, wrap the inner LLM so we log exact prompts/responses and pause between steps
    if headless:
        wrapped_llm = ChatOpenAI(model=model, temperature=0.0) if ChatOpenAI is not None else None
        if wrapped_llm is None:
            wrapped_llm = llm
        llm = LoggingLLM(wrapped_llm, log_folder=log_folder, pause_before_call=step_pause)
    env = NethackGymnasiumWrapper(render_mode='human' if not headless else 'rgb_array', des_file=des_file, autopickup=False)
    agent = create_llm_agent(env=env, llm=llm, memory_tokens=800, log_folder=log_folder, render=not headless)
    # Try to set map radius if available elsewhere
    try:
        agent.skill_selector.map_radius = 8
    except Exception:
        pass

    agent.init()
    try:
        task = input('Enter initial task for the agent (or blank for default): ')
        if task.strip() != '':
            agent.set_task(task)
        for step in agent.run():
            if step.thoughts:
                print(f"Thoughts: {step.thoughts}")
            # Do not log low-level executed actions here. interactive_llm
            # focuses only on LLM activity (prompts/responses) and
            # anything that may interrupt an action and force a rethink.
            if step.executed_action():
                # If the action produced events, print them since some
                # events (level change, teleport, new glyph, low health)
                # may cause the agent to stop executing a skill and call
                # the LLM again.
                if hasattr(step.step_data, 'events') and len(step.step_data.events) > 0:
                    try:
                        ev_msgs = [e.describe() for e in step.step_data.events]
                        print(f"Events (may trigger LLM rethink): {ev_msgs}")
                    except Exception:
                        # Fallback: print raw events
                        print(f"Events (may trigger LLM rethink): {step.step_data.events}")
            # No per-step pause: we only pause before LLM calls (handled by LoggingLLM.pause_before_call)
    except KeyboardInterrupt:
        print('Run interrupted by user')
    finally:
        agent.close()


def run_local_scenario(des_file: str, log_folder: str = None):
    """Run a local scenario using DummyLLM if available, otherwise InteractiveLLM.

    This mirrors the previous `tools/run_local_scenario.py` but consolidated here.
    """
    from netplay import create_llm_agent
    from netplay.nethack_utils.nle_wrapper import NethackGymnasiumWrapper
    llm = InteractiveLLM()

    if log_folder is None:
        log_folder = os.path.join(os.getcwd(), 'runs', 'local_test')
    os.makedirs(log_folder, exist_ok=True)

    env = NethackGymnasiumWrapper(render_mode='human', des_file=des_file, autopickup=False)
    agent = create_llm_agent(env=env, llm=llm, memory_tokens=500, log_folder=log_folder, render=False)
    agent.init()
    try:
        agent.set_task('Fulfill all your tasks in any order.')
        steps = 0
        for step in agent.run():
            if step.thoughts:
                print(f"Thoughts: {step.thoughts}")
            # As above: don't log every low-level action. Only show events
            # that could interrupt a running skill and require the LLM.
            if step.executed_action():
                if hasattr(step.step_data, 'events') and len(step.step_data.events) > 0:
                    try:
                        ev_msgs = [e.describe() for e in step.step_data.events]
                        print(f"Events (may trigger LLM rethink): {ev_msgs}")
                    except Exception:
                        print(f"Events (may trigger LLM rethink): {step.step_data.events}")
            steps += 1
            if steps >= 20:
                print('Reached test step limit, stopping')
                break
    finally:
        agent.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--des_file', default=None)
    parser.add_argument('--model', default='gpt-4-1106-preview')
    parser.add_argument('--log_folder', default=os.path.join(os.getcwd(), 'runs', 'interactive'))
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no render)')
    parser.add_argument('--step', action='store_true', help='Pause after each state; press Enter to continue')
    args = parser.parse_args()

    run_interactive(des_file=args.des_file, model=args.model, log_folder=args.log_folder, headless=args.headless or args.step, step_pause=args.step)

