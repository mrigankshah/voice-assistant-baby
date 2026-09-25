# Tool decision evaluation

This grades the first model decision, before app corrections. No tools were executed.
A pass for a no-tool case only means no tool was called; review the answer text separately.
Timings are complete first responses, including any model loading and thinking. They are not time to first spoken word.
Results use fixed test preferences and a fixed date, not your personal settings.

| Model | Decisions passed | Median | P95 |
| --- | --- | --- | --- |
| LiquidAI/lfm2.5-1.2b-instruct:latest | 0/5 | 7.75s | 9.44s |

## Failures

- **LiquidAI/lfm2.5-1.2b-instruct:latest / weather-default-today / repeat 1: wrong_tools**
  - User: What's the weather?
  - Expected: `[{"name": "get_weather", "arguments": {"day": "today"}}]`
  - Actual: `[{"name": "get_current_datetime", "arguments": {}}]`
- **LiquidAI/lfm2.5-1.2b-instruct:latest / weather-default-tomorrow / repeat 1: wrong_tools**
  - User: What's the weather tomorrow?
  - Expected: `[{"name": "get_weather", "arguments": {"day": "tomorrow"}}]`
  - Actual: `[{"name": "get_current_datetime", "arguments": {}}]`
- **LiquidAI/lfm2.5-1.2b-instruct:latest / weather-rain / repeat 1: missing_tool**
  - User: Will it rain tomorrow?
  - Expected: `[{"name": "get_weather", "arguments": {"day": "tomorrow"}}]`
  - Actual: `[]`
- **LiquidAI/lfm2.5-1.2b-instruct:latest / weather-explicit-london / repeat 1: wrong_tools**
  - User: What's the weather in London today?
  - Expected: `[{"name": "get_weather", "arguments": {"location": "London", "day": "today"}}]`
  - Actual: `[{"name": "get_current_datetime", "arguments": {}}]`
- **LiquidAI/lfm2.5-1.2b-instruct:latest / weather-explicit-sf / repeat 1: missing_tool**
  - User: Will it rain in San Francisco tomorrow?
  - Expected: `[{"name": "get_weather", "arguments": {"location": "San Francisco", "day": "tomorrow"}}]`
  - Actual: `[]`

Full prompts, responses, errors, and Ollama timings are in results.jsonl.
