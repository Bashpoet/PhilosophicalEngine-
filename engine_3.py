#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
philo_symphony_extended.py

A comprehensive script for orchestrating multi-round philosophical dialogues,
interdisciplinary discussions, and emergent pattern detection. It leverages
the Anthropic language model API to generate structured XML responses,
and includes:
    - Single-shot dialogues ("Philosopher's Stone")
    - Multi-turn dialogues
    - Advanced concept tracking via TF-IDF
    - Emergent pattern detection using a concept graph
    - Multiple dialectical frameworks: Hegelian (thesis/antithesis/synthesis) & Socratic
    - Configurable logging and system prompts
    - Demonstration of how to incorporate or mock user inputs, environment variables,
      or domain-specific knowledge

Requirements:
    - Python 3.8+
    - anthropic (pip install anthropic)
    - scikit-learn
    - networkx

Usage:
    1) Set ANTHROPIC_API_KEY in your environment:
       export ANTHROPIC_API_KEY="YOUR_API_KEY_HERE"
    2) python philo_symphony_extended.py
    3) Observe the printed logs and dialogues for each demonstration block.

Note:
    This file is intentionally verbose and exceeds 500 lines to illustrate
    a variety of possible expansions and deep integrations in a single file.
"""

import os
import re
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional

# External libraries for advanced features
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Attempt to import anthropic for LLM calls
try:
    from anthropic import Anthropic
except ImportError:
    raise ImportError("Please install anthropic with: pip install anthropic")

###############################################################################
# GLOBAL CONFIGURATION
###############################################################################

# Basic config for Python's logging library
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

LOGGER = logging.getLogger(__name__)

# We can load Anthropic API key from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    LOGGER.warning("No ANTHROPIC_API_KEY found in environment. "
                   "LLM calls will likely fail if you're making real requests.")

###############################################################################
# ANTHROPIC CLIENT SETUP
###############################################################################

# We'll attempt to instantiate the Anthropic client. If a fake key is used,
# actual calls will fail or return mock responses, but at least the code won't crash immediately.
client = Anthropic(api_key=ANTHROPIC_API_KEY)


###############################################################################
# HELPER FUNCTIONS: LLM CALLS, XML EXTRACTION, VALIDATION
###############################################################################

def llm_call(
    prompt: str,
    system_prompt: str = "",
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.4,
    messages: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Calls the Anthropic model with the given user prompt and (optionally) system prompt.
    You can optionally append this call to an existing conversation (messages).
    This function handles:
        - constructing the 'messages' format for Anthropic
        - capturing potential exceptions
        - returning the LLM's text

    Args:
        prompt (str): The user content for the model.
        system_prompt (str): The system-level instructions guiding the model's behavior.
        model (str): The ID or name of the model. Defaults to a hypothetical 'claude-3-5-sonnet-20241022'.
        temperature (float): Creativity parameter. Higher = more variety, lower = more deterministic.
        messages (List[Dict[str, str]]): The conversation so far, if doing multi-turn.

    Returns:
        str: The textual response from the model (or "" if error).
    """
    if messages is None:
        messages = []

    # Append the user's prompt as the last message
    messages.append({"role": "user", "content": prompt})

    LOGGER.debug("Calling LLM with system prompt: %s", system_prompt)
    LOGGER.debug("User prompt: %s", prompt)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            temperature=temperature,
        )
        # We'll assume the first chunk is the main text
        if response and len(response.content) > 0:
            ret_text = response.content[0].text
            LOGGER.debug("LLM response: %s", ret_text)
            return ret_text
        else:
            LOGGER.warning("No content returned from LLM response.")
            return ""
    except Exception as e:
        LOGGER.error("Error during llm_call: %s", e)
        return ""


def is_valid_xml(content: str) -> bool:
    """
    Tries to parse the given content as XML to see if it's well-formed.
    If parse fails, returns False.

    Args:
        content (str): The text to check.

    Returns:
        bool: True if valid XML, False otherwise.
    """
    try:
        ET.fromstring(content)
        return True
    except ET.ParseError:
        return False


def extract_xml(content: str, tag: str) -> str:
    """
    Extracts the content of the specified <tag> from the text using a simple Regex approach.
    If not found or doesn't match, returns "".

    Example:
        extract_xml("<root>some data</root>", "root") -> "some data"

    Args:
        content (str): The full text to search in.
        tag (str): The specific tag name or pattern, e.g., "root" or "Participant name='Plato'".

    Returns:
        str: The extracted substring, if found, else "".
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_xml_tag(content: str, tag: str) -> str:
    """
    Convenience alias for extract_xml, clarifies usage in some contexts.
    """
    return extract_xml(content, tag)


def extract_nested_tags(text: str, outer_tag: str, inner_tag: str) -> List[str]:
    """
    Extract all <inner_tag>...</inner_tag> segments from within <outer_tag>...</outer_tag>.

    Args:
        text (str): The text to parse.
        outer_tag (str): The parent tag name.
        inner_tag (str): The child tag name to find.

    Returns:
        List[str]: A list of all matching child content segments.
    """
    outer_content = extract_xml(text, outer_tag)
    if not outer_content:
        return []
    pattern = f"<{inner_tag}>(.*?)</{inner_tag}>"
    return re.findall(pattern, outer_content, re.DOTALL)


###############################################################################
# BASIC DIALOGUE FUNCTIONS (SINGLE-SHOT, MULTI-TURN)
###############################################################################

def generate_philosophers_dialogue(philosopher1: str, philosopher2: str, topic: str) -> None:
    """
    Single-shot 'Philosopher's Stone' scenario: prompts the LLM for a single dialogue
    with each philosopher labeled in XML. Then prints the results.

    Args:
        philosopher1 (str): Name of the first philosopher.
        philosopher2 (str): Name of the second philosopher.
        topic (str): The main topic of the conversation.
    """
    system_prompt = (
        f"You are the moderator of a conversation between {philosopher1} and {philosopher2}. "
        "Each speaks in their own style, referencing their distinct philosophies or ideas. "
        "Format the final answer using the following tags exactly:\n"
        f"<{philosopher1.lower()}> ... </{philosopher1.lower()}>\n"
        f"<{philosopher2.lower()}> ... </{philosopher2.lower()}>\n"
        "Do not provide any content outside these two tags."
    )
    user_prompt = f"The conversation topic is: {topic}. Engage in a robust exchange of ideas."

    raw_output = llm_call(user_prompt, system_prompt=system_prompt, temperature=0.5)

    # Minimal validation
    if not is_valid_xml(raw_output):
        LOGGER.warning("Response does not appear to be valid XML. Attempting single re-prompt.")
        regen_prompt = (
            "Your previous response was invalid XML. Please regenerate with the requested structure."
        )
        raw_output = llm_call(regen_prompt, system_prompt=system_prompt, temperature=0.5)

    speaker1_text = extract_xml_tag(raw_output, philosopher1.lower())
    speaker2_text = extract_xml_tag(raw_output, philosopher2.lower())

    print(f"=== Dialogue on '{topic}' ===\n")
    print(f"{philosopher1.upper()}:\n{speaker1_text}\n")
    print(f"{philosopher2.upper()}:\n{speaker2_text}\n")


def multi_turn_philosophers(philosopher1: str, philosopher2: str, topic: str, rounds: int = 3):
    """
    Multi-turn dialogue: each round extends the conversation with references to the prior text.
    The LLM is instructed to produce well-formed XML: <round> <philosopher1> ... </philosopher1> ...

    Args:
        philosopher1 (str): Name of the first philosopher.
        philosopher2 (str): Name of the second philosopher.
        topic (str): Topic to discuss.
        rounds (int): Number of iterative rounds.

    Returns:
        List[Dict[str, str]]: The conversation record (system + user + assistant messages).
    """
    system_prompt = (
        f"You are orchestrating a multi-round debate between {philosopher1} and {philosopher2} "
        f"on the topic '{topic}'. The format of each round must be:\n"
        f"<round>\n  <{philosopher1.lower()}>\n    ...\n  </{philosopher1.lower()}>\n"
        f"  <{philosopher2.lower()}>\n    ...\n  </{philosopher2.lower()}>\n</round>\n"
        "No text should appear outside these tags. Make sure it's valid XML. "
        "Keep each philosopher's style distinct."
    )

    conversation = [{"role": "system", "content": system_prompt}]

    for current_round in range(1, rounds + 1):
        user_prompt = (
            f"This is round {current_round}. Each philosopher should respond in turn, "
            f"building on prior statements about '{topic}'."
        )
        raw_output = llm_call(
            prompt=user_prompt,
            system_prompt=system_prompt,
            messages=conversation,
            temperature=0.7
        )

        if not is_valid_xml(raw_output):
            LOGGER.warning("Round %d: invalid XML. Attempting re-prompt.", current_round)
            regen_prompt = (
                "Your previous response was invalid. Please follow the exact structure:\n"
                f"<round><{philosopher1.lower()}>...</{philosopher1.lower()}><{philosopher2.lower()}>"
                f"...</{philosopher2.lower()}></round>"
            )
            raw_output = llm_call(
                prompt=regen_prompt,
                system_prompt=system_prompt,
                messages=conversation,
                temperature=0.7
            )

        conversation.append({"role": "assistant", "content": raw_output})
        print(f"\n===== ROUND {current_round} =====\n")
        print(raw_output)

    return conversation


###############################################################################
# SYMPHONY OF SYNTHESIS (MULTIPLE EXPERTS)
###############################################################################

def symphony_of_synthesis(fields: List[str], topic: str) -> None:
    """
    Asks multiple domain experts to comment on a shared topic. 
    Each domain is labeled in XML with <expert domain='...'> tags.

    Args:
        fields (List[str]): e.g. ["Baroque Music", "Software Architecture", "Mycology"]
        topic (str): The topic they should unify around.
    """
    domain_tags = []
    for field in fields:
        safe_field = field.lower().replace(" ", "_")
        domain_tags.append(f"<expert domain='{safe_field}'> ... </expert>")

    system_prompt = (
        "You are orchestrating a roundtable among diverse domain experts. "
        "They must respond in distinct voices, focusing on potential overlaps and synergies. "
        "Format the final answer using the following tags exactly:\n"
        + "\n".join(domain_tags)
        + "\nNo extra text outside these tags."
    )
    user_prompt = (
        f"Each expert should explore how '{topic}' relates to their domain, then highlight surprising connections."
    )

    raw_output = llm_call(user_prompt, system_prompt=system_prompt, temperature=0.6)

    if not is_valid_xml(raw_output):
        LOGGER.warning("Symphony initial response invalid. Attempting re-prompt.")
        regen_prompt = (
            "Your previous response was invalid XML. Please follow the structure exactly."
        )
        raw_output = llm_call(regen_prompt, system_prompt=system_prompt, temperature=0.6)

    print(f"=== SYMPHONY OF SYNTHESIS ON: {topic} ===\n")
    for field in fields:
        safe_field = field.lower().replace(" ", "_")
        block = extract_xml_tag(raw_output, f"expert domain='{safe_field}'")
        print(f"--- Insights from {field} ---\n{block}\n")


###############################################################################
# ADVANCED CONCEPT TRACKING (TF-IDF)
###############################################################################

def advanced_track_concepts(
    conversation_history: List[str],
    concept_dictionary: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Tracks concept relevance using TF-IDF across the entire conversation history.

    Steps:
        1) Convert each round or segment in conversation_history into a doc in a corpus.
        2) Fit a TfidfVectorizer, ignoring English stopwords.
        3) For each concept in concept_dictionary, accumulate the TF-IDF for any keywords that appear.
        4) Return a dict mapping each concept to a cumulative weight.

    Args:
        conversation_history (List[str]): List of textual segments from the conversation.
        concept_dictionary (Dict[str, List[str]]): Mapping from concept labels to keywords.

    Returns:
        Dict[str, float]: Concept weights, e.g., {"Recursion": 2.0, "Allegory of the Cave": 1.3, ...}
    """
    if not conversation_history:
        return {}

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(conversation_history)
    feature_names = vectorizer.get_feature_names_out()

    concept_weights = {concept: 0.0 for concept in concept_dictionary}

    for row_idx in range(X.shape[0]):
        row_vec = X.getrow(row_idx).toarray().flatten()
        for concept, keywords in concept_dictionary.items():
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in feature_names:
                    kw_idx = np.where(feature_names == kw_lower)[0]
                    if kw_idx.size > 0:
                        # Accumulate TF-IDF
                        concept_weights[concept] += row_vec[kw_idx[0]]

    return concept_weights


###############################################################################
# BUILDING A CONCEPT GRAPH & DETECTING EMERGENT PATTERNS
###############################################################################

def build_concept_graph(
    conversation_history: List[str],
    concept_dictionary: Dict[str, List[str]]
) -> nx.Graph:
    """
    Creates a NetworkX graph where each node is a concept. We add or strengthen edges
    if multiple concepts appear in the same sentence. Edge weights represent the
    co-occurrence frequency.

    Args:
        conversation_history (List[str]): List of textual segments (e.g., each round).
        concept_dictionary (Dict[str, List[str]]): {concept_label: [keywords]}

    Returns:
        nx.Graph: A graph with nodes = concepts, edges = synergy. Edge attribute "weight" is co-occurrence count.
    """
    G = nx.Graph()
    for c in concept_dictionary.keys():
        G.add_node(c)

    for segment in conversation_history:
        sentences = re.split(r'[.!?]', segment)
        for sentence in sentences:
            present_concepts = []
            for concept, keywords in concept_dictionary.items():
                if any(kw.lower() in sentence.lower() for kw in keywords):
                    present_concepts.append(concept)
            # Form edges among the concepts that appear together
            for i in range(len(present_concepts)):
                for j in range(i + 1, len(present_concepts)):
                    c1, c2 = present_concepts[i], present_concepts[j]
                    if G.has_edge(c1, c2):
                        G[c1][c2]['weight'] += 1
                    else:
                        G.add_edge(c1, c2, weight=1)
    return G


def detect_emergent_patterns(concept_graph: nx.Graph) -> List[str]:
    """
    A naive approach to detecting "emergent" patterns: we look for edges with weight >=2
    or connected components of size >2.

    Args:
        concept_graph (nx.Graph): A graph built by build_concept_graph.

    Returns:
        List[str]: A list of textual descriptions of potential emergent synergies.
    """
    insights = []

    # Edge-based synergy
    for (u, v, data) in concept_graph.edges(data=True):
        weight = data.get('weight', 0)
        if weight >= 2:
            synergy_msg = f"Synergy: '{u}' and '{v}' co-occurred with weight={weight}."
            insights.append(synergy_msg)

    # Connected components
    for component in nx.connected_components(concept_graph):
        if len(component) > 2:
            c_list = sorted(list(component))
            msg = f"Multi-concept synergy among: {', '.join(c_list)}"
            insights.append(msg)

    return insights


###############################################################################
# MULTIPLE DIALECTICAL STRUCTURES (HEGELIAN & SOCRATIC)
###############################################################################

def build_system_prompt_for_hegelian(participants: List[str], round_num: int, history: List[str]) -> str:
    """
    Constructs a system prompt to produce <Thesis>, <Antithesis>, <Synthesis> for each participant.

    Args:
        participants (List[str]): The thinkers or participants in the debate.
        round_num (int): The current round index, for reference in the prompt.
        history (List[str]): List of prior LLM outputs or partial conversation.

    Returns:
        str: A system-level instruction string for the LLM.
    """
    participant_blocks = []
    for p in participants:
        safe_p = p.replace(" ", "_").lower()
        block = (
f"<Participant name='{safe_p}'>\n"
"    <Thesis>...</Thesis>\n"
"    <Antithesis>...</Antithesis>\n"
"    <Synthesis>...</Synthesis>\n"
"</Participant>"
        )
        participant_blocks.append(block)

    prompt = (
        f"You are orchestrating round {round_num+1} in a Hegelian dialectic among: "
        + ", ".join(participants)
        + ". Use the following structure for each participant:\n\n"
        + "\n".join(participant_blocks)
        + "\n\nPrior discussion:\n"
        + "\n".join(history)
        + "\nDo not include content outside these tags."
    )
    return prompt


def build_system_prompt_for_socratic(participants: List[str], round_num: int, history: List[str]) -> str:
    """
    Constructs a system prompt to produce <Question>, <AttemptedDefinition>, <CounterExample>, <Refinement>.

    Args:
        participants (List[str]): The thinkers or participants in the debate.
        round_num (int): Current round index.
        history (List[str]): Past conversation or outputs.

    Returns:
        str: A system-level instruction for Socratic structure.
    """
    participant_blocks = []
    for p in participants:
        safe_p = p.replace(" ", "_").lower()
        block = (
f"<Participant name='{safe_p}'>\n"
"    <Question>...</Question>\n"
"    <AttemptedDefinition>...</AttemptedDefinition>\n"
"    <CounterExample>...</CounterExample>\n"
"    <Refinement>...</Refinement>\n"
"</Participant>"
        )
        participant_blocks.append(block)

    prompt = (
        f"You are orchestrating round {round_num+1} in a Socratic dialogue among: "
        + ", ".join(participants)
        + ". Use the following structure for each participant:\n\n"
        + "\n".join(participant_blocks)
        + "\n\nPrior conversation:\n"
        + "\n".join(history)
        + "\nNo extra text outside these tags."
    )
    return prompt


def parse_hegelian_round(response: str, participants: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Extracts <Thesis>, <Antithesis>, <Synthesis> from each participant's block.

    Args:
        response (str): The raw text produced by the LLM for a given round.
        participants (List[str]): The participants to look for.

    Returns:
        Dict[str, Dict[str, str]]: For each participant, a dict of {"Thesis": "...", "Antithesis": "...", "Synthesis": "..."}.
    """
    results = {}
    for p in participants:
        safe_p = p.replace(" ", "_").lower()
        p_block = extract_xml(response, f"Participant name='{safe_p}'")
        if p_block:
            thesis = extract_xml(p_block, "Thesis")
            antithesis = extract_xml(p_block, "Antithesis")
            synthesis = extract_xml(p_block, "Synthesis")
            results[p] = {
                "Thesis": thesis,
                "Antithesis": antithesis,
                "Synthesis": synthesis
            }
    return results


def parse_socratic_round(response: str, participants: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Extracts <Question>, <AttemptedDefinition>, <CounterExample>, <Refinement> from each participant.

    Args:
        response (str): The LLM's output for the round.
        participants (List[str]): Names of participants.

    Returns:
        Dict[str, Dict[str, str]]: For each participant, a dict of the Socratic steps.
    """
    results = {}
    for p in participants:
        safe_p = p.replace(" ", "_").lower()
        p_block = extract_xml(response, f"Participant name='{safe_p}'")
        if p_block:
            question = extract_xml(p_block, "Question")
            definition = extract_xml(p_block, "AttemptedDefinition")
            counterex = extract_xml(p_block, "CounterExample")
            refinement = extract_xml(p_block, "Refinement")
            results[p] = {
                "Question": question,
                "AttemptedDefinition": definition,
                "CounterExample": counterex,
                "Refinement": refinement
            }
    return results


def run_dialectical_debate(
    participants: List[str],
    topic: str,
    rounds: int = 2,
    style: str = "hegelian",
    temperature: float = 0.7
) -> List[str]:
    """
    Orchestrates a multi-round dialectical debate. style can be "hegelian" or "socratic".

    For each round:
        1) Build a system prompt based on style.
        2) Attempt to parse the result, printing out each participant's phases.

    Args:
        participants (List[str]): The participants in the debate.
        topic (str): The topic of discussion.
        rounds (int): Number of iterative rounds.
        style (str): "hegelian" or "socratic".
        temperature (float): Creativity setting for LLM.

    Returns:
        List[str]: The raw LLM outputs for each round.
    """
    conversation_history: List[str] = []
    for r in range(rounds):
        if style.lower() == "hegelian":
            system_prompt = build_system_prompt_for_hegelian(participants, r, conversation_history)
        else:
            system_prompt = build_system_prompt_for_socratic(participants, r, conversation_history)

        user_prompt = f"Round {r+1} about '{topic}' in {style} style."
        response = llm_call(prompt=user_prompt, system_prompt=system_prompt, temperature=temperature)

        if not is_valid_xml(response):
            LOGGER.warning("Response invalid for round %d. Attempting re-prompt.", r+1)
            regen_prompt = (
                f"Your previous response was invalid for a {style} dialogue. Please regenerate with correct structure."
            )
            response = llm_call(prompt=regen_prompt, system_prompt=system_prompt, temperature=temperature)

        conversation_history.append(response)

        # Parse & display
        if style.lower() == "hegelian":
            round_data = parse_hegelian_round(response, participants)
            print(f"\n=== Hegelian Round {r+1} ===\n")
            for p, phases in round_data.items():
                print(
                    f"{p}:\n  Thesis: {phases['Thesis']}\n"
                    f"         Antithesis: {phases['Antithesis']}\n"
                    f"         Synthesis: {phases['Synthesis']}\n"
                )
        else:
            round_data = parse_socratic_round(response, participants)
            print(f"\n=== Socratic Round {r+1} ===\n")
            for p, phases in round_data.items():
                print(
                    f"{p}:\n  Question: {phases['Question']}\n"
                    f"  AttemptedDefinition: {phases['AttemptedDefinition']}\n"
                    f"  CounterExample: {phases['CounterExample']}\n"
                    f"  Refinement: {phases['Refinement']}\n"
                )

    return conversation_history


###############################################################################
# DEMO / MAIN
###############################################################################

def interactive_loop():
    """
    An optional interactive loop that asks users to input a style, participants, etc.,
    then runs a short dialogue. This is purely illustrative, showing how one might
    incorporate user-driven flows in a console environment.
    """
    print("Welcome to the Philosophical Engine Interactive Loop!")
    print("You can type 'exit' at any prompt to quit.\n")

    while True:
        style = input("Choose a dialogue style (hegelian / socratic / exit): ").strip().lower()
        if style == "exit":
            break
        if style not in ["hegelian", "socratic"]:
            print("Unknown style. Try again or type 'exit' to quit.")
            continue

        participants_input = input("Enter participants (comma-separated), e.g. 'Plato, Nietzsche': ")
        if participants_input.strip().lower() == "exit":
            break
        participants = [p.strip() for p in participants_input.split(",") if p.strip()]

        topic_input = input("Enter a topic to discuss (or 'exit'): ")
        if topic_input.strip().lower() == "exit":
            break

        rounds_input = input("How many rounds? (default 2): ")
        if rounds_input.strip().lower() == "exit":
            break
        try:
            rounds_val = int(rounds_input.strip()) if rounds_input.strip() else 2
        except ValueError:
            rounds_val = 2

        print(f"\nInitiating a {style} dialogue among {participants} about '{topic_input}'.\n")
        run_dialectical_debate(participants, topic_input, rounds=rounds_val, style=style)


def main_demo():
    """
    This function runs a series of demonstration calls for the various features.
    The final code block in __main__ calls this function.
    """
    # 1) Single-Shot Philosophers
    print("=== SINGLE-SHOT PHILOSOPHERS DEMO ===")
    generate_philosophers_dialogue("Plato", "Douglas_Hofstadter", "the nature of consciousness")

    # 2) Multi-Turn Philosophers
    print("\n=== MULTI-TURN PHILOSOPHERS DEMO ===")
    multi_turn_philosophers("Plato", "Nietzsche", "the will to truth", rounds=2)

    # 3) Symphony of Synthesis
    print("\n=== SYMPHONY OF SYNTHESIS DEMO ===")
    symphony_of_synthesis(["Baroque Music", "Software Architecture", "Mycology"], "emergent complexity")

    # 4) Advanced Concept Tracking + Emergent Pattern Detection
    print("\n=== ADVANCED CONCEPT TRACKING & EMERGENT PATTERN DETECTION DEMO ===")

    concept_dict = {
        "Forms": ["forms", "platonic", "ideal"],
        "Recursion": ["recursion", "self-reference", "loop"],
        "Allegory of the Cave": ["cave", "shadows", "prisoners"],
        "Emergent Complexity": ["emergent", "complexity", "emergence"]
    }

    # Some mock text for demonstration
    conversation_history_mock = [
        "Plato introduced the concept of forms and also the famous cave allegory about shadows.",
        "Nietzsche responded with a challenge to ideal forms, but recursion was also discussed in terms of self-reference.",
        "They ended up referencing emergent complexity in the interplay of ideas regarding illusions and loops."
    ]

    concept_weights = advanced_track_concepts(conversation_history_mock, concept_dict)
    print("Concept Weights (TF-IDF Approximation):")
    for c, w in concept_weights.items():
        print(f"  {c}: {w:.3f}")

    G = build_concept_graph(conversation_history_mock, concept_dict)
    emergent = detect_emergent_patterns(G)
    print("\nEmergent Patterns or Synergies:")
    for synergy in emergent:
        print(f"  {synergy}")

    # 5) Dialectical Debate: Hegelian
    print("\n=== HEGELIAN DIALECTICAL DEBATE DEMO ===")
    run_dialectical_debate(["Aristotle", "Kant"], "metaphysics of ethics", rounds=2, style="hegelian")

    # 6) Dialectical Debate: Socratic
    print("\n=== SOCRATIC DIALOGUE DEMO ===")
    run_dialectical_debate(["Socrates", "Descartes"], "the nature of knowledge", rounds=2, style="socratic")


if __name__ == "__main__":
    # We’ll do a standard set of demos. If you want an interactive console approach,
    # uncomment the interactive_loop() line below.
    # interactive_loop()

    main_demo()
