# AMD395 × HelloAgents Hands-on: Smart Travel Planner

This tutorial demonstrates how to build a local smart travel-planning agent on AMD AI PC using:

- Local LLM inference (Ollama / LM Studio / Linglong AI app)
- MCP tool-calling protocol
- AMap APIs for attractions and weather

## Goals

1. Run an LLM locally on AMD hardware
2. Connect MCP tools to map/weather APIs
3. Generate practical travel itinerary in Markdown

## Prerequisites

- Python 3.10+
- AMD ROCm-compatible runtime environment
- A local model endpoint
- AMap API key

## Setup

```shell
python -m pip install --upgrade pip
pip install hello-agents requests python-dotenv uv
```

Then either:

1. set `amap_api_key` directly in `travel_planner_mcp.py`, or
2. export the API key through an environment variable:

```shell
setx AMAP_MAPS_API_KEY "your_amap_api_key_here"
```

## Run

```shell
python travel_planner_mcp.py
```

The sample script currently runs with a built-in Hangzhou example. Update the values in `main()` if you want a different destination, number of days, budget, or preferences. The agent will:

1. query POIs and weather
2. reason over budget constraints
3. produce a day-by-day travel plan

## Output

The generated result is saved as Markdown (example: [hangzhou-3-day-mcp.md](../examples/hangzhou-3-day-mcp.md)).

## Architecture Notes

- Agent core: HelloAgents
- Tool bridge: `uvx amap-mcp-server`
- External service: AMap API
- Runtime: local OpenAI-compatible endpoint

## Notes

The Chinese tutorial is available as [amd395-helloagents-smart-travel-planner-zh.md](./amd395-helloagents-smart-travel-planner-zh.md).

