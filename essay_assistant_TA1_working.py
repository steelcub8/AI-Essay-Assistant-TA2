from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_ollama import ChatOllama


# ============================================================
# STATE - Using LangGraph MessagesState
# ============================================================

class EssayState(MessagesState):
    topic: str
    draft: str
    critique: str
    issues_found: bool
    revised_text: str
    demo_no_issues: bool


# ============================================================
# LOCAL LLM
# ============================================================

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)


# ============================================================
# NODE 1 - DRAFT
# ============================================================

def draft_node(state: EssayState):

    topic = state["topic"]

    prompt = f"""
Write a short student-level paragraph of around 80 words
on this topic:

{topic}

Make it clear, relevant and suitable for a college assignment.
Return ONLY the paragraph.
"""

    response = llm.invoke(prompt)

    draft = response.content.strip()

    print("\n" + "=" * 60)
    print("NODE 1: DRAFT")
    print("=" * 60)
    print(draft)

    return {
        "draft": draft
    }


# ============================================================
# NODE 2 - CRITIQUE
# ============================================================

def critique_node(state: EssayState):

    draft = state["draft"]

    # --------------------------------------------------------
    # TRACE 2 DEMONSTRATION
    # --------------------------------------------------------
    # This deliberately demonstrates the "no issues" path.
    # The graph routing itself still depends on issues_found.
    # --------------------------------------------------------

    if state.get("demo_no_issues", False):

        critique = "NO_ISSUES"
        issues_found = False

    else:

        prompt = f"""
You are the CRITIQUE stage of a writing assistant.

Review this student paragraph:

--- DRAFT ---
{draft}
--- END DRAFT ---

Check for:
1. Grammar problems
2. Clarity problems
3. Relevance problems
4. Organization problems

IMPORTANT:
If you find ANY problem, you MUST list the problems as:

1. Problem...
2. Problem...
3. Problem...

Do NOT write NO_ISSUES if you have identified any problem.

Only write:

NO_ISSUES

when there are genuinely no significant problems.

Return ONLY the critique.
"""

        response = llm.invoke(prompt)

        critique = response.content.strip()

        # ----------------------------------------------------
        # Robust issue detection
        # ----------------------------------------------------

        if critique.upper().strip() == "NO_ISSUES":

            issues_found = False

        else:

            issues_found = True

    print("\n" + "=" * 60)
    print("NODE 2: CRITIQUE")
    print("=" * 60)
    print(critique)

    print("\nIssues Found:", issues_found)

    return {
        "critique": critique,
        "issues_found": issues_found
    }


# ============================================================
# CONDITIONAL EDGE
# ============================================================

def route_after_critique(state: EssayState):

    if state["issues_found"]:

        print("\n>>> CONDITIONAL EDGE")
        print("Issues found -> Going to REVISE")

        return "revise"

    else:

        print("\n>>> CONDITIONAL EDGE")
        print("No issues found -> Going directly to END")

        return END


# ============================================================
# NODE 3 - REVISE
# ============================================================

def revise_node(state: EssayState):

    draft = state["draft"]
    critique = state["critique"]

    prompt = f"""
You are the REVISE stage of a writing assistant.

Original Draft:
{draft}

Critique:
{critique}

Rewrite the paragraph and fix all the identified problems.

Keep the original meaning and topic.
Improve grammar, clarity and organization.

Return ONLY the revised paragraph.
"""

    response = llm.invoke(prompt)

    revised_text = response.content.strip()

    print("\n" + "=" * 60)
    print("NODE 3: REVISE")
    print("=" * 60)
    print(revised_text)

    return {
        "revised_text": revised_text
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

builder = StateGraph(EssayState)


# Add nodes
builder.add_node("draft", draft_node)
builder.add_node("critique", critique_node)
builder.add_node("revise", revise_node)


# START -> DRAFT
builder.add_edge(START, "draft")


# DRAFT -> CRITIQUE
builder.add_edge("draft", "critique")


# CRITIQUE -> CONDITIONAL ROUTING
builder.add_conditional_edges(
    "critique",
    route_after_critique,
    {
        "revise": "revise",
        END: END
    }
)


# REVISE -> END
builder.add_edge("revise", END)


# Compile graph
graph = builder.compile()


# ============================================================
# RUN DEMO
# ============================================================

def run_demo(topic, demo_no_issues=False):

    print("\n\n")
    print("#" * 70)

    if demo_no_issues:
        print("TRACE 2: REVISION NOT NEEDED")
    else:
        print("TRACE 1: REVISION NEEDED")

    print("#" * 70)

    print("\nTopic:", topic)

    initial_state = {
        "messages": [],
        "topic": topic,
        "draft": "",
        "critique": "",
        "issues_found": False,
        "revised_text": "",
        "demo_no_issues": demo_no_issues
    }

    final_state = graph.invoke(initial_state)

    print("\n" + "#" * 70)
    print("FINAL RESULT")
    print("#" * 70)

    if final_state.get("revised_text"):

        print("\nRevision WAS required.")

        print("\nFinal Revised Paragraph:")
        print(final_state["revised_text"])

    else:

        print("\nRevision was NOT required.")

        print("\nFinal Paragraph:")
        print(final_state["draft"])


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # TRACE 1 - ISSUES FOUND
    # ========================================================

    run_demo(
        "Importance of Artificial Intelligence in Education",
        demo_no_issues=False
    )

    # ========================================================
    # TRACE 2 - NO ISSUES FOUND
    # ========================================================

    run_demo(
        "Benefits of Regular Exercise",
        demo_no_issues=True
    )