import streamlit as st
import workflow  # Backend workflow module

st.set_page_config(page_title="LangGraph Prompt Chain", layout="wide")
st.title("Dynamic LangGraph Prompt Executor")

# Initialize session state for dynamic prompt list
if "prompts" not in st.session_state:
    st.session_state.prompts = [""]

def add_prompt():
    st.session_state.prompts.append("")

def remove_prompt():
    if len(st.session_state.prompts) > 1:
        st.session_state.prompts.pop()

# Buttons to add/remove prompts dynamically
col1, col2 = st.columns(2)
with col1:
    st.button("Add Prompt", on_click=add_prompt)
with col2:
    st.button("Remove Prompt", on_click=remove_prompt)

st.write("Enter each prompt below. Each prompt can have multiple lines.")

# Render input fields for each prompt
for i in range(len(st.session_state.prompts)):
    st.session_state.prompts[i] = st.text_area(
        label=f"Prompt {i+1}",
        value=st.session_state.prompts[i],
        key=f"prompt_{i}",
        height=150  # Allow larger area for multiline text
    )

# Run the workflow
if st.button("Run Workflow"):
    prompts = [p.strip() for p in st.session_state.prompts if p.strip()]
    
    if not prompts:
        st.warning("Please enter at least one prompt.")
    else:
        with st.spinner("Executing workflow..."):
            final_output = workflow.run_prompt_workflow(prompts)
        
        st.subheader("Final Output")
        st.text_area("Output", value=final_output, height=400)
