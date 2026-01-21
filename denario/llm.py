from pydantic import BaseModel
from typing import Dict

class LLM(BaseModel):
    """LLM base model"""
    name: str
    """Name/identifier of the model."""
    max_output_tokens: int
    """Maximum output tokens allowed."""
    temperature: float | None
    """Temperature of the model."""

gemini20flash = LLM(name="gemini-2.0-flash",
                    max_output_tokens=8192,
                    temperature=0.7)
"""`gemini-2.0-flash` model."""

gemini25flash = LLM(name="gemini-2.5-flash",
                    max_output_tokens=65536,
                    temperature=0.7)
"""`gemini-2.5-flash` model."""

gemini25pro = LLM(name="gemini-2.5-pro",
                  max_output_tokens=65536,
                  temperature=0.7)
"""`gemini-2.5-pro` model."""

o3mini = LLM(name="o3-mini-2025-01-31",
             max_output_tokens=100000,
             temperature=None)
"""`o3-mini` model."""

gpt4o = LLM(name="gpt-4o-2024-11-20",
            max_output_tokens=16384,
            temperature=0.5)
"""`gpt-4o` model."""

gpt41 = LLM(name="gpt-4.1-2025-04-14",
            max_output_tokens=16384,
            temperature=0.5)
"""`gpt-4.1` model."""

gpt41mini = LLM(name="gpt-4.1-mini",
                max_output_tokens=16384,
                temperature=0.5)
"""`gpt-4.1-mini` model."""

gpt4omini = LLM(name="gpt-4o-mini-2024-07-18",
                max_output_tokens=16384,
                temperature=0.5)
"""`gpt-4o-mini` model."""

gpt45 = LLM(name="gpt-4.5-preview-2025-02-27",
            max_output_tokens=16384,
            temperature=0.5)
"""`gpt-4.5-preview` model."""

gpt5 = LLM(name="gpt-5",
           max_output_tokens=128000,
           temperature=None)
"""`gpt-5` model """

gpt5mini = LLM(name="gpt-5-mini",
               max_output_tokens=128000,
               temperature=None)
"""`gpt-5-mini` model."""

claude37sonnet = LLM(name="claude-3-7-sonnet-20250219",
                     max_output_tokens=64000,
                     temperature=0)
"""`claude-3-7-sonnet` model."""

claude4opus = LLM(name="claude-opus-4-20250514",
                   max_output_tokens=32000,
                   temperature=0)
"""`claude-4-Opus` model."""

claude41opus = LLM(name="claude-opus-4-1-20250805",
                   max_output_tokens=32000,
                   temperature=0)
"""`claude-4.1-Opus` model."""

# Ollama models (local, free)
qwen25coder32b = LLM(name="qwen2.5-coder:32b",
                     max_output_tokens=32768,
                     temperature=0.7)
"""`qwen2.5-coder:32b` model - Best local coding model."""

qwen25coder14b = LLM(name="qwen2.5-coder:14b",
                     max_output_tokens=32768,
                     temperature=0.7)
"""`qwen2.5-coder:14b` model - Good local coding model."""

qwen25coder7b = LLM(name="qwen2.5-coder:7b",
                    max_output_tokens=32768,
                    temperature=0.7)
"""`qwen2.5-coder:7b` model - Fast local coding model."""

qwen2572b = LLM(name="qwen2.5:72b",
                max_output_tokens=32768,
                temperature=0.7)
"""`qwen2.5:72b` model - Best local general model."""

qwen2532b = LLM(name="qwen2.5:32b",
                max_output_tokens=32768,
                temperature=0.7)
"""`qwen2.5:32b` model - Great local general model."""

qwen2514b = LLM(name="qwen2.5:14b",
                max_output_tokens=32768,
                temperature=0.7)
"""`qwen2.5:14b` model - Good local general model."""

qwen257b = LLM(name="qwen2.5:7b",
               max_output_tokens=32768,
               temperature=0.7)
"""`qwen2.5:7b` model - Fast local general model."""

# Qwen 3 models (latest generation)
qwen3coder30b = LLM(name="qwen3-coder:30b",
                    max_output_tokens=32768,
                    temperature=0.7)
"""`qwen3-coder:30b` model - Latest generation coding model."""

qwen330b = LLM(name="qwen3:30b",
               max_output_tokens=32768,
               temperature=0.7)
"""`qwen3:30b` model - Latest generation general model."""

llama3370b = LLM(name="llama3.3:70b",
                 max_output_tokens=32768,
                 temperature=0.7)
"""`llama3.3:70b` model - Excellent local model."""

llama33text = LLM(name="llama3.3:text",
                  max_output_tokens=32768,
                  temperature=0.7)
"""`llama3.3:text` model - Optimized llama3.3."""

llama323b = LLM(name="llama3.2:3b",
                max_output_tokens=32768,
                temperature=0.7)
"""`llama3.2:3b` model - Fast local reasoning model."""

deepseekcoderv216b = LLM(name="deepseek-coder-v2:16b",
                         max_output_tokens=32768,
                         temperature=0.7)
"""`deepseek-coder-v2:16b` model - Good local coding model."""

deepseekmathv3 = LLM(name="deepseek-math:7b",
                     max_output_tokens=32768,
                     temperature=0.7)
"""`deepseek-math:7b` model - Math reasoning model."""

models : Dict[str, LLM] = {
                            "gemini-2.0-flash" : gemini20flash,
                            "gemini-2.5-flash" : gemini25flash,
                            "gemini-2.5-pro" : gemini25pro,
                            "o3-mini" : o3mini,
                            "gpt-4o" : gpt4o,
                            "gpt-4.1" : gpt41,
                            "gpt-4.1-mini" : gpt41mini,
                            "gpt-4o-mini" : gpt4omini,
                            "gpt-4.5" : gpt45,
                            "gpt-5" : gpt5,
                            "gpt-5-mini" : gpt5mini,
                            "claude-3.7-sonnet" : claude37sonnet,
                            "claude-4-opus" : claude4opus,
                            "claude-4.1-opus" : claude41opus,
                            # Ollama models (local, free)
                            "qwen2.5-coder:32b" : qwen25coder32b,
                            "qwen2.5-coder:14b" : qwen25coder14b,
                            "qwen2.5-coder:7b" : qwen25coder7b,
                            "qwen2.5:72b" : qwen2572b,
                            "qwen2.5:32b" : qwen2532b,
                            "qwen2.5:14b" : qwen2514b,
                            "qwen2.5:7b" : qwen257b,
                            # Qwen 3 models (latest generation)
                            "qwen3-coder:30b" : qwen3coder30b,
                            "qwen3:30b" : qwen330b,
                            "llama3.3:70b" : llama3370b,
                            "llama3.3:text" : llama33text,
                            "llama3.2:3b" : llama323b,
                            "deepseek-coder-v2:16b" : deepseekcoderv216b,
                            "deepseek-math:7b" : deepseekmathv3,
                           }
"""Dictionary with the available models."""
