# app.py
# Streamlit + LangGraph sequential prompt chaining with per-prompt .docx upload support
#
# Requirements:
#   pip install streamlit python-docx python-dotenv langchain langgraph openai
#
# Usage:
#   1) Create a .env file with: OPENAI_API_KEY=sk-...
#   2) streamlit run app.py

import os
from dotenv import load_dotenv
from typing import List, TypedDict, Optional, Any
import io

import streamlit as st
from docx import Document

# langchain Chat model import — support multiple possible locations across versions
try:
    # Preferred import for many langchain versions
    from langchain.chat_models import ChatOpenAI
except Exception:
    try:
        # Some distributions expose an `openai` submodule
        from langchain.chat_models.openai import ChatOpenAI
    except Exception:
        ChatOpenAI = None

from langgraph.graph import StateGraph, START, END

# -----------------------------
# Load API key from .env file
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -----------------------------
# State definition
# -----------------------------
class ChainState(TypedDict):
    text: str
    step_outputs: List[str]


# -----------------------------
# Helpers
# -----------------------------
def read_docx_file(uploaded_file) -> str:
    """
    Read a .docx file (streamlit UploadedFile) and return its text content.
    """
    try:
        file_bytes = uploaded_file.read()
        document = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in document.paragraphs if p.text]
        return "\n\n".join(paragraphs).strip()
    except Exception as e:
        st.error(f"Failed to read uploaded .docx: {e}")
        return ""


def make_llm_node(prompt_template: str, doc_text_for_step: Optional[str], llm: Any):
    """
    Create a LangGraph node function for a single prompt.
    Behavior regarding doc_text_for_step:
      - If doc_text_for_step is not empty: the node will use that content as the primary
        `{text}` when rendering the prompt for this step (overriding prior state.text).
      - If it's empty/None: the node will use state['text'] as `{text}`.
    The node also has access to `{steps}` which is the concatenation of previous step outputs.
    """
    def node(state: ChainState) -> dict:
        prev_text = state.get("text", "")
        prev_steps = state.get("step_outputs", []) or []

        # Decide what to use as {text}
        if doc_text_for_step and doc_text_for_step.strip():
            input_text = doc_text_for_step
        else:
            input_text = prev_text

        # Render prompt
        rendered = prompt_template.format(
            text=input_text,
            steps="\n\n".join(prev_steps) if prev_steps else ""
        )

        # Call LLM using a compatible method across different wrappers
        raw_response = None
        # try common sync call methods
        if hasattr(llm, "predict"):
            raw_response = llm.predict(rendered)
        elif hasattr(llm, "invoke"):
            raw_response = llm.invoke(rendered)
        elif hasattr(llm, "run"):
            raw_response = llm.run(rendered)
        elif callable(llm):
            raw_response = llm(rendered)
        else:
            raise RuntimeError("LLM object has no supported call method (predict/invoke/run/__call__)")

        # Normalize response to text
        if isinstance(raw_response, str):
            output_text = raw_response
        else:
            # Common wrapper shapes: objects with .content or .text
            if hasattr(raw_response, "content"):
                output_text = getattr(raw_response, "content")
            elif hasattr(raw_response, "text"):
                output_text = getattr(raw_response, "text")
            else:
                # Fallback to string representation
                output_text = str(raw_response)

        # Update state: new 'text' becomes this response; append to previous step_outputs
        return {
            "text": output_text,
            "step_outputs": state.get("step_outputs", []) + [output_text],
        }

    return node


def build_and_invoke(prompts: List[str], docs_for_steps: List[Optional[str]], initial_text: str,
                     model_name: str = "gpt-3.5-turbo", temperature: float = 0.0):
    """
    Build a LangGraph sequential workflow dynamically from prompts and optional doc texts,
    invoke it with the initial state, and return the final state.
    docs_for_steps is a list aligned with prompts; each element is either the text content
    extracted from an uploaded .docx for that step or None/""
    """
    # Initialize LLM client
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not found in environment (.env).")

    # Create LLM client in a way that supports multiple langchain/langchain-provider layouts.
    if ChatOpenAI is not None:
        # Try to instantiate the provider-specific ChatOpenAI directly.
        try:
            llm = ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=OPENAI_API_KEY)
        except Exception:
            try:
                llm = ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=OPENAI_API_KEY)
            except Exception:
                # Fallback to the unified factory which requires the provider integration package
                from langchain.chat_models import init_chat_model

                model_str = model_name if ":" in model_name else f"openai:{model_name}"
                llm = init_chat_model(model_str, temperature=temperature, openai_api_key=OPENAI_API_KEY)
    else:
        # Use langchain's init_chat_model factory (this will require the provider package, e.g. langchain-openai)
        from langchain.chat_models import init_chat_model

        model_str = model_name if ":" in model_name else f"openai:{model_name}"
        llm = init_chat_model(model_str, temperature=temperature, openai_api_key=OPENAI_API_KEY)

    builder = StateGraph(ChainState)

    node_names = []
    for i, prompt in enumerate(prompts):
        node_name = f"step_{i}"
        doc_text = docs_for_steps[i] if i < len(docs_for_steps) else None
        node_fn = make_llm_node(prompt, doc_text, llm)
        builder.add_node(node_name, node_fn)
        node_names.append(node_name)

    if len(node_names) == 0:
        # trivial graph: passthrough
        def nop(state: ChainState) -> dict:
            return {}
        builder.add_node("noop", nop)
        builder.add_edge(START, "noop")
        builder.add_edge("noop", END)
    else:
        # Wire sequential edges
        builder.add_edge(START, node_names[0])
        for a, b in zip(node_names, node_names[1:]):
            builder.add_edge(a, b)
        builder.add_edge(node_names[-1], END)

    # Compile and run
    graph = builder.compile()
    initial_state: ChainState = {"text": initial_text, "step_outputs": []}
    result = graph.invoke(initial_state)
    return result


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="LangGraph + .docx per prompt", layout="wide")
st.title("LangGraph Sequential Prompt Chain — Upload .docx per Prompt")

st.markdown(
    """
This app lets you edit a list of prompts and optionally upload a `.docx` file for **each** prompt.
If a .docx is uploaded for a step, that document's text will be used as the `{text}` when rendering
the prompt for that step (overriding the previous step's output). If no .docx is uploaded, the chain
uses the previous step's output (or the initial text for step 1).
"""
)

# Initialize session state structures
if "prompts" not in st.session_state:
    st.session_state.prompts = [
        "Summarize the following text: {text}",
        "Rewrite the summary in a sarcastic tone.",
        "Translate the rewritten text into Spanish."
    ]
if "uploaded_docs" not in st.session_state:
    # store None or the text content extracted from uploaded .docx
    st.session_state.uploaded_docs = [None for _ in st.session_state.prompts]
if "model_name" not in st.session_state:
    st.session_state.model_name = "gpt-3.5-turbo"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.0

left_col, right_col = st.columns([2, 3])

with left_col:
    st.header("Prompts & Uploads (per step)")
    # Ensure uploaded_docs length matches prompts
    while len(st.session_state.uploaded_docs) < len(st.session_state.prompts):
        st.session_state.uploaded_docs.append(None)
    while len(st.session_state.uploaded_docs) > len(st.session_state.prompts):
        st.session_state.uploaded_docs.pop()

    for idx in range(len(st.session_state.prompts)):
        st.subheader(f"Step {idx+1}")
        pcol1, pcol2, pcol3 = st.columns([6, 1, 1])
        with pcol1:
            st.session_state.prompts[idx] = st.text_area(f"Prompt {idx+1}", value=st.session_state.prompts[idx], key=f"prompt_{idx}", height=100)
        with pcol2:
            if st.button("↑", key=f"up_{idx}"):
                if idx > 0:
                    st.session_state.prompts[idx - 1], st.session_state.prompts[idx] = st.session_state.prompts[idx], st.session_state.prompts[idx - 1]
                    st.session_state.uploaded_docs[idx - 1], st.session_state.uploaded_docs[idx] = st.session_state.uploaded_docs[idx], st.session_state.uploaded_docs[idx - 1]
                    st.experimental_rerun()
        with pcol3:
            if st.button("↓", key=f"down_{idx}"):
                if idx < len(st.session_state.prompts) - 1:
                    st.session_state.prompts[idx + 1], st.session_state.prompts[idx] = st.session_state.prompts[idx], st.session_state.prompts[idx + 1]
                    st.session_state.uploaded_docs[idx + 1], st.session_state.uploaded_docs[idx] = st.session_state.uploaded_docs[idx], st.session_state.uploaded_docs[idx + 1]
                    st.experimental_rerun()

        # File uploader for this step
        uploaded = st.file_uploader(f"Upload .docx for Step {idx+1} (optional)", type=["docx"], key=f"file_{idx}")
        if uploaded is not None:
            text_content = read_docx_file(uploaded)
            if text_content:
                st.session_state.uploaded_docs[idx] = text_content
                st.markdown("**Document preview:**")
                st.write(text_content[:1000] + ("..." if len(text_content) > 1000 else ""))
            else:
                st.warning("Uploaded .docx could not be read or is empty.")
        else:
            # Show current stored doc text (if any) and option to clear
            if st.session_state.uploaded_docs[idx]:
                with st.expander("Uploaded document text (clear to remove)"):
                    st.write(st.session_state.uploaded_docs[idx][:1000] + ("..." if len(st.session_state.uploaded_docs[idx]) > 1000 else ""))
                    if st.button(f"Clear uploaded doc for step {idx+1}", key=f"clear_doc_{idx}"):
                        st.session_state.uploaded_docs[idx] = None
                        st.experimental_rerun()

        st.markdown("---")

    # Add / remove prompts
    add_col, rem_col = st.columns([1, 1])
    with add_col:
        if st.button("➕ Add Prompt"):
            st.session_state.prompts.append("New prompt... Use {text} and {steps}")
            st.session_state.uploaded_docs.append(None)
            st.experimental_rerun()
    with rem_col:
        if st.button("➖ Remove Last Prompt"):
            if st.session_state.prompts:
                st.session_state.prompts.pop()
                st.session_state.uploaded_docs.pop()
                st.experimental_rerun()

    st.markdown("**Prompt template variables:** `{text}` = current input for this step (doc override or previous output), `{steps}` = all previous step outputs joined.")

    st.markdown("---")
    st.markdown("**Model settings**")
    st.text_input("Model name", value=st.session_state.model_name, key="model_name_input")
    st.number_input("Temperature", value=float(st.session_state.temperature), min_value=0.0, max_value=1.0, step=0.01, key="temperature_input")
    st.session_state.model_name = st.session_state.get("model_name_input") or st.session_state.model_name
    st.session_state.temperature = float(st.session_state.get("temperature_input", st.session_state.temperature))

with right_col:
    st.header("Run the Chain")
    st.text_area("Initial input text (used when a step has no uploaded doc and is the first step)", value="Paste your initial text here...", key="initial_input", height=200)
    initial_text = st.session_state.get("initial_input", "")

    st.markdown("**OpenAI API Key Source**")
    st.write("This app reads OPENAI_API_KEY from the `.env` file in the same folder. Current status:")
    if OPENAI_API_KEY:
        st.success("OPENAI_API_KEY loaded from .env")
    else:
        st.error("OPENAI_API_KEY not found in .env. Please add OPENAI_API_KEY=sk-... to a .env file.")

    if st.button("▶️ Run Sequential Chain"):
        # Validation
        if any((p.strip() == "" for p in st.session_state.prompts)):
            st.error("All prompts must be non-empty.")
        else:
            try:
                with st.spinner("Running the LangGraph pipeline..."):
                    final_state = build_and_invoke(
                        prompts=st.session_state.prompts,
                        docs_for_steps=st.session_state.uploaded_docs,
                        initial_text=initial_text,
                        model_name=st.session_state.model_name,
                        temperature=st.session_state.temperature,
                    )
                st.success("Chain completed.")
                st.header("Final Output (state['text'])")
                st.code(final_state.get("text", ""), language="text")

                st.header("Intermediate Step Outputs")
                for i, s in enumerate(final_state.get("step_outputs", []) or []):
                    with st.expander(f"Step {i+1} output", expanded=(i == len(final_state.get("step_outputs", [])) - 1)):
                        st.write(s)

            except Exception as exc:
                st.exception(f"Error running pipeline: {exc}")

st.sidebar.header("Notes")
st.sidebar.markdown(
    """
- If you upload a `.docx` for a step, its text **overrides** the chain's previous text for that step and is used as `{text}`.
- If you don't upload a `.docx` for a step, the step uses the prior step's output (or the initial input for the first step).
- Use `{steps}` in a prompt to access all previous step outputs joined together.
- Ensure `.env` with `OPENAI_API_KEY` is present in the same folder as this app.
"""
)

###.\.venv\Scripts\Activate    
##pip install -r requirements.txt
 #run app.py --server.port 8501