import os
from dotenv import load_dotenv
# Import ChatOpenAI in a way that's compatible across langchain versions:
try:
    # Newer setups may provide the provider integration as a separate package
    from langchain.chat_models import ChatOpenAI
except Exception:
    try:
        # Provider integration package commonly exposes ChatOpenAI here
        from langchain_openai import ChatOpenAI
    except Exception:
        ChatOpenAI = None

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing import List, TypedDict, Any

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM initialization — create a chat model in a way that works across installs
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment (.env).")

if ChatOpenAI is not None:
    # Try common constructor signatures; tolerate different kwarg names
    try:
        llm = ChatOpenAI(model_name="gpt-5.1", temperature=0,  reasoning_effort="high",openai_api_key=OPENAI_API_KEY)
    except Exception:
        try:
            llm = ChatOpenAI(model="gpt-5.1", temperature=0, reasoning_effort="high", openai_api_key=OPENAI_API_KEY)
        except Exception:
            # Fall back to factory
            llm = init_chat_model("openai:gpt-5.1", temperature=0,  reasoning_effort="high",openai_api_key=OPENAI_API_KEY)
else:
    # Use the unified factory which will require the provider integration to be installed
    llm = init_chat_model("openai:gpt-3.5-turbo", temperature=0,  reasoning_effort="high",openai_api_key=OPENAI_API_KEY)

# Define the state model for LangGraph as a TypedDict
class PromptState(TypedDict):
    context: str


def _call_llm_sync(llm: Any, prompt_text: str) -> str:
    """Call llm with a prompt and return the text result (supports multiple APIs)."""
    raw = None
    if hasattr(llm, "predict"):
        raw = llm.predict(prompt_text)
    elif hasattr(llm, "invoke"):
        raw = llm.invoke(prompt_text)
    elif hasattr(llm, "run"):
        raw = llm.run(prompt_text)
    elif callable(llm):
        raw = llm(prompt_text)
    else:
        raise RuntimeError("LLM object has no supported call method (predict/invoke/run/__call__)")

    if isinstance(raw, str):
        return raw
    if hasattr(raw, "content"):
        return getattr(raw, "content")
    if hasattr(raw, "text"):
        return getattr(raw, "text")
    return str(raw)


def run_prompt_workflow(prompts: List[str]) -> str:
    """
    Builds and executes a LangGraph sequential workflow for N prompts.
    Returns the final cumulative context as a string.
    """
    # Build a StateGraph where each node is a function that updates the state dict
    builder = StateGraph(PromptState)

    node_names: List[str] = []

    def make_node_fn(prompt_text: str):
        def node_fn(state: PromptState) -> dict:
            prev_context = state.get("context", "")
            final_prompt = f"{prompt_text}\n\nPrevious Outputs:\n{prev_context}"
            response_text = _call_llm_sync(llm, final_prompt)
            new_context = prev_context + f"\n\n=== Output Step ===\n{response_text}"
            return {"context": new_context}

        return node_fn

    for i, p in enumerate(prompts):
        node_name = f"step_{i}"
        builder.add_node(node_name, make_node_fn(p))
        node_names.append(node_name)

    if node_names:
        builder.add_edge(START, node_names[0])
        for a, b in zip(node_names, node_names[1:]):
            builder.add_edge(a, b)
        builder.add_edge(node_names[-1], END)
    else:
        # trivial passthrough
        def noop(state: PromptState) -> dict:
            return {"context": state.get("context", "")}

        builder.add_node("noop", noop)
        builder.add_edge(START, "noop")
        builder.add_edge("noop", END)

    graph = builder.compile()
    initial_state: PromptState = {"context": ""}
    result = graph.invoke(initial_state)
    return result.get("context", "")
