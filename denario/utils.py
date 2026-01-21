import os
import re
from pathlib import Path
import warnings

from .llm import LLM, models

def input_check(str_input: str) -> str:
    """Check if the input is a string with the desired content or the path markdown file, in which case reads it to get the content."""

    if str_input.endswith(".md"):
        with open(str_input, 'r') as f:
            content = f.read()
    elif isinstance(str_input, str):
        content = str_input
    else:
        raise ValueError("Input must be a string or a path to a markdown file.")
    return content

def llm_parser(llm: LLM | str) -> LLM:
    """Get the LLM instance from a string."""

    if isinstance(llm, str):
        try:
            llm = models[llm]
        except KeyError:
            raise KeyError(f"LLM '{llm}' not available. Please select from: {list(models.keys())}")
    return llm

def extract_file_paths(markdown_text):
    """
    Extract the bulleted file paths from markdown text 
    and check if they exist and are absolute paths.
    
    Args:
        markdown_text (str): The markdown text containing file paths
    
    Returns:
        tuple: (existing_paths, missing_paths)
    """
    
    # Pattern to match file paths in markdown bullet points
    pattern = r'-\s*([^\n]+\.(?:csv|txt|md|py|json|yaml|yml|xml|html|css|js|ts|tsx|jsx|java|cpp|c|h|hpp|go|rs|php|rb|pl|sh|bat|sql|log))'
    
    # Find all matches
    matches = re.findall(pattern, markdown_text, re.IGNORECASE)
    
    # Clean up paths and check existence
    existing_paths = []
    missing_paths = []
    
    for match in matches:
        path = match.strip()
        if os.path.exists(path) and os.path.isabs(path):
            existing_paths.append(path)
        else:
            missing_paths.append(path)
    
    return existing_paths, missing_paths

def check_file_paths(content: str) -> None:
    """Check that file paths indicated in content text have the proper format"""

    existing_paths, missing_paths = extract_file_paths(content)

    if len(missing_paths) > 0:
        warnings.warn(
            f"The following data files paths in the data description are not in the right format or do not exist:\n"
            f"{missing_paths}\n"
            f"Please fix them according to the convention '- /absolute/path/to/file.ext'\n"
            f"otherwise this may cause hallucinations in the LLMs."
        )

    if len(existing_paths) == 0:
        warnings.warn(
            "No data files paths were found in the data description. If you want to provide input data, ensure that you indicate their path, otherwise this may cause hallucinations in the LLM in the get_results() workflow later on."
        )

def create_work_dir(work_dir: str | Path, name: str) -> Path:
    """Create working directory"""

    work_dir = os.path.join(work_dir, f"{name}_generation_output")
    os.makedirs(work_dir, exist_ok=True)
    return Path(work_dir)

def get_task_result(chat_history, name: str):
    """Get task result from chat history
    
    For cmbagent mode, the actual results are stored in messages from 'plan_recorder' or formatted agents.
    The chat history has a specific structure where the final plan is recorded by plan_recorder.
    
    Due to cmbagent routing issues, we may need to search more broadly for results.
    """
    
    # For cmbagent idea generation, look for 'plan_recorder' instead of 'idea_maker_nest' 
    if name == 'idea_maker_nest':
        # Try to find the actual result in various possible locations
        # 1. First try plan_recorder
        result = _search_chat_history(chat_history, 'plan_recorder')
        if result:
            return result
        
        # 2. Try idea_maker_response_formatter
        result = _search_chat_history(chat_history, 'idea_maker_response_formatter')
        if result:
            return result
            
        # 3. Try the last substantive message from idea_maker or idea_hater
        for obj in chat_history[::-1]:
            if not isinstance(obj, dict):
                continue
            if obj.get('name') in ['idea_maker', 'idea_hater']:
                content = obj.get('content', '')
                # Check if this looks like a final research idea (has title or description)
                if content and len(content) > 200 and ('title' in content.lower() or 'research' in content.lower()):
                    return content
        
        # 4. Fall through to standard search
        name = 'plan_recorder'
    
    # For cmbagent method/experiment generation, look for researcher output
    elif name == 'researcher_response_formatter':
        # Try to find the researcher output
        # 1. First try the formatter itself
        result = _search_chat_history(chat_history, 'researcher_response_formatter')
        if result:
            return result
        
        # 2. Try just 'researcher'
        result = _search_chat_history(chat_history, 'researcher')
        if result:
            return result
            
        # 3. Try the last substantive message from researcher
        for obj in chat_history[::-1]:
            if not isinstance(obj, dict):
                continue
            if obj.get('name') == 'researcher':
                content = obj.get('content', '')
                # Check if this looks like methodology or results
                if content and len(content) > 200:
                    return content
    
    result = _search_chat_history(chat_history, name)
    
    if result is None:
        raise ValueError(f"No result found for task '{name}' in chat history")
    
    return result


def _search_chat_history(chat_history, agent_name: str):
    """Helper function to search for agent output in chat history"""
    for obj in chat_history[::-1]:
        # Skip objects that aren't dicts
        if not isinstance(obj, dict):
            continue
            
        # Check for messages without a name but with substantial content (>10000 chars)
        # These often contain the final formatted output
        if 'name' not in obj or not obj.get('name'):
            if 'content' in obj and isinstance(obj['content'], str) and len(obj['content']) > 10000:
                return obj['content']
            continue
            
        if obj['name'] == agent_name:
            # Get content - might be in 'content' field or in tool_calls
            if 'content' in obj and obj['content']:
                return obj['content']
            # Check if there are tool_calls with arguments containing the result
            elif 'tool_calls' in obj and obj['tool_calls']:
                for tool_call in obj['tool_calls']:
                    if 'function' in tool_call and 'arguments' in tool_call['function']:
                        import json
                        try:
                            args = json.loads(tool_call['function']['arguments'])
                            # For plan_recorder, look for 'plan_suggestion'
                            if 'plan_suggestion' in args:
                                return args['plan_suggestion']
                            # For idea recording, look for 'improved_main_task'
                            elif 'improved_main_task' in args:
                                return args['improved_main_task']
                        except:
                            pass
    return None

def in_notebook():
    """Check whether the code is run from a Jupyter Notebook or not, to use different display options"""
    
    try:
        from IPython import get_ipython # type: ignore
        if 'IPKernelApp' not in get_ipython().config:  # type: ignore # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True
