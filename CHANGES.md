# Changes for Ollama Integration

## Motivation

This fork adds support for running Denario entirely with local Ollama models, eliminating dependency on external API services (OpenAI, Google, Anthropic). This enables:

- **100% local execution** - No data leaves your machine
- **Zero API costs** - No per-token charges
- **Privacy** - Sensitive research data remains private
- **Offline capability** - Work without internet connectivity
- **Custom models** - Use any Ollama-compatible model

## Modified Files

### `denario/utils.py`

**Changes:**
- Enhanced `get_task_result()` function to handle CMBAgent's autogen chat history format
- Added flexible result extraction with multiple fallback strategies
- Added `_search_chat_history()` helper function for robust message parsing

**Motivation:**
CMBAgent (built on autogen) stores chat history differently than Denario expected. The original code looked for agent name `'idea_maker_nest'` but autogen stores results under `'plan_recorder'` or in tool_call JSON arguments. The enhanced function:

1. Maps agent names correctly (`'idea_maker_nest'` → `'plan_recorder'`)
2. Parses tool_calls JSON to extract results from function arguments
3. Falls back to searching multiple agent types if primary search fails
4. Searches for substantive content from any relevant agent

This robust approach ensures Denario can extract research ideas even when CMBAgent's agent routing doesn't follow the expected pattern.

**Code Details:**
```python
# Before: Simple name-based lookup that failed with autogen format
for obj in chat_history:
    if obj['name'] == name:
        return obj['content']

# After: Multi-strategy extraction with JSON parsing and fallbacks
- Check plan_recorder for formatted plans
- Parse tool_calls JSON arguments (plan_suggestion, improved_main_task)
- Search idea_maker_response_formatter
- Fall back to substantive messages from idea_maker/idea_hater
```

## Compatibility

These changes maintain backward compatibility with the original Denario API. All modifications are in internal result extraction logic and do not affect the public interface.

## Testing

Successfully tested with:
- Ollama models: qwen3:30b, qwen3-coder:30b, llama3.3:70b
- Hardware: NVIDIA A40 GPU (48GB VRAM)
- Dataset: Euclid SPV slitless spectroscopy (768 FITS images)
- Workflow: Full 7-step pipeline (idea → literature → methods → experiments → paper → referee)

## Benefits

- **Reliability**: Flexible extraction handles CMBAgent agent routing variations
- **Robustness**: Multiple fallback strategies prevent workflow failures
- **Maintainability**: Clear separation between search logic and result extraction
- **Debuggability**: Helper function makes testing individual search strategies easier

## Future Work

- Consider upstreaming these improvements to handle autogen chat formats natively
- Add configuration option to specify preferred agent name mappings
- Enhance error messages with debugging information about available agents

## Related Changes

See `cmbagent/CHANGES.md` for CMBAgent modifications that complement these changes.
