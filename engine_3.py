#!/usr/bin/env python3
"""
philo_symphony_extended.py

A comprehensive script that demonstrates:
1. Multi-turn philosophical dialogues with structured XML output and validation.
2. Interdisciplinary discussions ("Symphony of Synthesis").
3. Advanced concept tracking using TF-IDF.
4. Emergent pattern detection using a naive graph-based approach.
5. Multiple dialectical structures (Hegelian, Socratic).

Dependencies:
    pip install anthropic
    pip install scikit-learn
    pip install networkx
    (and Python 3.8+ recommended)

To run:
    1) Set ANTHROPIC_API_KEY in your environment:
       export ANTHROPIC_API_KEY="your_key"
    2) python philo_symphony_extended.py
"""

import os
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Any

# External libraries for advanced features
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Anthropic library for LLM calls
try:
    from anthropic import Anthropic
except ImportError:
    raise ImportError("Please install anthropic with: pip install anthropic")

###############################################################################
# ANTHROPIC CLIENT SETUP
###############################################################################

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "FAKE_KEY_FOR_DEMO"))

###############################################################################
# CORE LLM CALL AND XML UTILITIES
###############################################################################

def llm_call(
    prompt: str,
    system_prompt: str = "",
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.4,
    messages: List[Dict[str, str]] = None
) -> str:
    """
    Calls the Anthropic model with the given prompt (and optional system_prompt)
    to get a response. Optionally appends the prompt to an existing conversation
    (messages), allowing for multi-turn continuity.
    """
    if messages is None:
        messages = []

    # The user's prompt is appended as the last message in the conversation
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            temperature=temperature,
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error during llm_call: {e}")
        return ""

def is_valid_xml(content: str) -> bool:
    """
    Tries to parse the given content as XML. Returns True if valid, False otherwise.
    """
    try:
        ET.fromstring(content)
        return True
    except ET.ParseError:
        return False

def extract_xml(content: str, tag: str) -> str:
    """
    Extracts the content of the specified XML tag from the given text.
    Returns an empty string if the tag is not found.
    Uses a simple pattern-based approach (Regex).
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_xml_tag(content: str, tag: str) -> str:
    """
    Similar to extract_xml, but provided for naming clarity. Returns the first match only.
    """
    return extract_xml(content, tag)

def extract_nested_tags(text: str, outer_tag: str, inner_tag: str) -> List[str]:
    """
    Returns a list of all <inner_tag>...</inner_tag> found within the <outer_tag>...</outer_tag> block.
    """
    outer_content = extract_xml(text, outer_tag)
    if not outer_content:
        return []
    pattern = f"<{inner_tag}>(.*?)</{inner_tag}>"
    return re.findall(pattern, outer_content, re.DOTALL)


###############################################################################
# SINGLE-SHOT AND MULTI-TURN DIALOGUES
###############################################################################

def generate_philosophers_dialogue(philosopher1: str, philosopher2: str, topic: str) -> None:
    """
    Implements a single-shot 'Philosopher's Stone' scenario:
    a dialogue between philosopher1 and philosopher2 about the given topic,
    with each philosopher's statement wrapped in XML tags for easy parsing.
    """
    system_prompt = (
        f"You are the moderator of a conversation between {philosopher1} and {philosopher2}. "
        f"Each speaks in their own style, referencing their distinct philosophies or ideas. "
        "Format the final answer using the following tags exactly:\n"
        f"<{philosopher1.lower()}> ... </{philosopher1.lower()}>\n"
        f"<{philosopher2.lower()}> ... </{philosopher2.lower()}>\n"
        "Do not provide any content outside these two tags."
    )
    user_prompt = (
        f"The conversation topic is: {topic}. Engage in a robust exchange of ideas, each philosopher responding in turn."
    )
    raw_output = llm_call(user_prompt, system_prompt=system_prompt, temperature=0.5)

    # Attempt minimal validation, and if invalid, ask to regenerate once
    if not is_valid_xml(raw_output):
        regen_prompt = (
            "Your previous response was not valid XML or did not match the structure. "
            "Please regenerate with the exact tag structure requested."
        )
        raw_output = llm_call(regen_prompt, system_prompt=system_prompt, temperature=0.5)

    # Extract each philosopher's speech
    philosopher1_text = extract_xml_tag(raw_output, philosopher1.lower())
    philosopher2_text = extract_xml_tag(raw_output, philosopher2.lower())

    print(f"=== Dialogue on {topic} ===\n")
    print(f"{philosopher1.upper()}:\n{philosopher1_text}\n")
    print(f"{philosopher2.upper()}:\n{philosopher2_text}\n")


def multi_turn_philosophers(philosopher1: str, philosopher2: str, topic: str, rounds: int = 3):
    """
    Demonstrates a multi-turn dialogue between two philosophers about a specified topic.
    Each round is an iterative call. We try to ensure well-formed XML each time.
    We'll store the conversation so far in 'conversation' to preserve context.
    """
    system_prompt = f"""You are orchestrating a multi-round debate between {philosopher1} and {philosopher2} on the topic "{topic}".
The format of each round must be:
<round>
  <{philosopher1.lower()}>
    Their statement goes here
  </{philosopher1.lower()}>
  <{philosopher2.lower()}>
    Their statement goes here
  </{philosopher2.lower()}>
</round>
Do not include anything outside these XML tags. Make sure it's valid XML. 
Keep the style distinct for each philosopher."""

    conversation = [
        {"role": "system", "content": system_prompt}
    ]

    for current_round in range(1, rounds + 1):
        user_prompt = (
            f"This is round {current_round}. Each philosopher should respond in turn, "
            "building on what was said previously. "
            f"The topic is still: {topic}."
        )
        raw_output = llm_call(
            prompt=user_prompt,
            system_prompt=system_prompt,
            messages=conversation,
            temperature=0.7
        )

        # Validate
        if not is_valid_xml(raw_output):
            regen_prompt = (
                "Your previous response was not valid XML or incorrect. "
                f"Please regenerate with the exact tag structure: "
                f"<round><{philosopher1.lower()}>...</{philosopher1.lower()}><{philosopher2.lower()}>...</{philosopher2.lower()}></round>."
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
# "SYMPHONY OF SYNTHESIS" - INTERDISCIPLINARY EXPERTS
###############################################################################

def symphony_of_synthesis(fields: List[str], topic: str) -> None:
    """
    Implements the 'Symphony of Synthesis' by asking multiple domain experts
    to discuss how their domains intersect around a given topic, each labeled with XML.
    """
    domain_tags = []
    for field in fields:
        safe_field = field.lower().replace(" ", "_")
        domain_tags.append(f"<expert domain='{safe_field}'> ... </expert>")

    system_prompt = (
        "You are orchestrating a roundtable among diverse domain experts. "
        "They must respond in distinct voices, focusing on potential overlaps and synergies. "
        "Format your final answer using the following tags exactly, and do not provide text outside them:\n\n"
        + "\n".join(domain_tags)
        + "\n"
    )
    user_prompt = f"Each expert should explore how {topic} relates to their domain, then highlight surprising connections."

    raw_output = llm_call(user_prompt, system_prompt=system_prompt, temperature=0.6)

    # Attempt minimal validation
    if not is_valid_xml(raw_output):
        regen_prompt = (
            "Your previous response was not valid XML or did not match the structure. "
            "Regenerate using the exact tag structure requested."
        )
        raw_output = llm_call(regen_prompt, system_prompt=system_prompt, temperature=0.6)

    print(f"=== SYMPHONY OF SYNTHESIS ON: {topic} ===\n")
    for field in fields:
        safe_field = field.lower().replace(" ", "_")
        content = extract_xml_tag(raw_output, f"expert domain='{safe_field}'")
        print(f"--- Insights from {field} ---\n{content}\n")


###############################################################################
# ADVANCED CONCEPT TRACKING (TF-IDF) & EMERGENT PATTERN DETECTION
###############################################################################

def advanced_track_concepts(
    conversation_history: List[str],
    concept_dictionary: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Tracks concept relevance using TF-IDF across the entire conversation.
    conversation_history: a list of strings (text output from each round).
    concept_dictionary: a dict mapping concept labels to sets/lists of keywords.

    Returns a dict: {concept: cumulative_weight}
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Each round is one doc
    corpus = conversation_history
    if not corpus:
        return {}

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)  # shape: [num_rounds, num_features]
    feature_names = vectorizer.get_feature_names_out()

    concept_weights = {concept: 0.0 for concept in concept_dictionary}

    # For each round, identify top terms and see if they map to known concepts
    for row_idx in range(X.shape[0]):
        round_vec = X.getrow(row_idx).toarray().flatten()
        # For speed, gather an index for all features with nonzero tf-idf
        nonzero_indices = round_vec.nonzero()[0]

        # Check each concept's keywords
        for concept, keywords in concept_dictionary.items():
            for kw in keywords:
                # If the keyword is actually in the vocabulary
                if kw.lower() in feature_names:
                    kw_idx = np.where(feature_names == kw.lower())[0]
                    if kw_idx.size > 0 and kw_idx[0] in nonzero_indices:
                        concept_weights[concept] += round_vec[kw_idx[0]]

    return concept_weights

def build_concept_graph(
    conversation_history: List[str],
    concept_dictionary: Dict[str, List[str]]
) -> nx.Graph:
    """
    Creates a graph where nodes are concepts, edges occur when two or more concepts
    appear in the same sentence or round.
    """
    G = nx.Graph()
    # Initialize nodes
    for c in concept_dictionary.keys():
        G.add_node(c)

    # For each chunk of text, we do naive sentence splits
    for round_text in conversation_history:
        sentences = re.split(r'[.!?]', round_text)
        for sentence in sentences:
            present_concepts = []
            for concept, keywords in concept_dictionary.items():
                # If any keyword is in that sentence
                if any(kw.lower() in sentence.lower() for kw in keywords):
                    present_concepts.append(concept)

            # Form edges among present_concepts
            for i in range(len(present_concepts)):
                for j in range(i+1, len(present_concepts)):
                    c1, c2 = present_concepts[i], present_concepts[j]
                    if G.has_edge(c1, c2):
                        G[c1][c2]['weight'] += 1
                    else:
                        G.add_edge(c1, c2, weight=1)
    return G

def detect_emergent_patterns(concept_graph: nx.Graph) -> List[str]:
    """
    Naively checks for edges of higher weight or newly formed cliques 
    that might indicate emergent synergy. Returns a list of textual insights.
    """
    emergent_insights = []

    # Example: look for edges above a certain threshold
    for (u, v, w) in concept_graph.edges(data=True):
        if w.get('weight', 0) >= 2:
            # We consider this a synergy
            msg = f"Emergent synergy detected between '{u}' and '{v}' (weight={w['weight']})."
            emergent_insights.append(msg)

    # Optional: find cliques or connected components with > 2 concepts
    for component in nx.connected_components(concept_graph):
        if len(component) > 2:
            emergent_insights.append(
                f"Multi-concept synergy among: {', '.join(component)}"
            )

    return emergent_insights

###############################################################################
# MULTIPLE DIALECTICAL STRUCTURES (HEGELIAN, SOCRATIC)
###############################################################################

def build_system_prompt_for_hegelian(participants: List[str], round_num: int, history: List[str]) -> str:
    """
    Instructs each participant to produce <Thesis>, <Antithesis>, <Synthesis> tags.
    """
    participant_tags = []
    for p in participants:
        safe_p = p.replace(" ", "_").lower()
        participant_tags.append(
f"""<Participant name='{safe_p}'>
    <Thesis>...</Thesis>
    <Antithesis>...</Antithesis>
    <Synthesis>...</Synthesis>
</Participant>"""
        )

    prompt = (
        f"You are orchestrating round {round_num+1} in a Hegelian dialectic among: "
        + ", ".join(participants)
        + ". Use the following structure for each participant:\n\n"
        + "\n".join(participant_tags)
        + "\n\nPrior conversation:\n"
        + "\n".join(history)
        + "\nNo extra text outside these XML tags."
    )
    return prompt

def build_system_prompt_for_socratic(participants: List[str], round_num: int, history: List[str]) -> str:
    """
    Instructs each participant to produce <Question>, <AttemptedDefinition>, <CounterExample>, <Refinement>.
    """
    participant_tags = []
    for p in participants:
        safe_p = p.replace(" ", "_").lower()
        participant_tags.append(
f"""<Participant name='{safe_p}'>
    <Question>...</Question>
    <AttemptedDefinition>...</AttemptedDefinition>
    <CounterExample>...</CounterExample>
    <Refinement>...</Refinement>
</Participant>"""
        )

    prompt = (
        f"You are orchestrating round {round_num+1} in a Socratic dialogue among: "
        + ", ".join(participants)
        + ". Use the following structure for each participant:\n\n"
        + "\n".join(participant_tags)
        + "\n\nPrior conversation:\n"
        + "\n".join(history)
        + "\nNo extra text outside these XML tags."
    )
    return prompt

def parse_hegelian_round(response: str, participants: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Extracts <Thesis>, <Antithesis>, <Synthesis> from each participant.
    """
    results = {}
    for p in participants:
        safe_p = p.replace(" ", "_").lower()
        block = extract_xml(response, f"Participant name='{safe_p}'")
        if block:
            thesis = extract_xml(block, "Thesis")
            antithesis = extract_xml(block, "Antithesis")
            synthesis = extract_xml(block, "Synthesis")
            results[p] = {
                "Thesis": thesis,
                "Antithesis": antithesis,
                "Synthesis": synthesis
            }
    return results

def parse_socratic_round(response: str, participants: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Extracts <Question>, <AttemptedDefinition>, <CounterExample>, <Refinement> from each participant.
    """
    results = {}
    for p in participants:
        safe_p = p.replace(" ", "_").lower()
        block = extract_xml(response, f"Participant name='{safe_p}'")
        if block:
            question = extract_xml(block, "Question")
            definition = extract_xml(block, "AttemptedDefinition")
            counter_example = extract_xml(block, "CounterExample")
            refinement = extract_xml(block, "Refinement")
            results[p] = {
                "Question": question,
                "AttemptedDefinition": definition,
                "CounterExample": counter_example,
                "Refinement": refinement
            }
    return results


###############################################################################
# A GENERAL "DIALECTICAL ENGINE" SHOWING Hegelian OR Socratic ROUNDS
###############################################################################

def run_dialectical_debate(
    participants: List[str],
    topic: str,
    rounds: int = 2,
    style: str = "hegelian"
) -> List[str]:
    """
    style can be "hegelian" or "socratic".
    Each round constructs a different system prompt based on style.
    """
    conversation_history = []
    for r in range(rounds):
        if style.lower() == "hegelian":
            system_prompt = build_system_prompt_for_hegelian(participants, r, conversation_history)
        else:
            system_prompt = build_system_prompt_for_socratic(participants, r, conversation_history)

        user_prompt = f"Round {r+1} on '{topic}' in {style} style. Continue the dialogue."

        response = llm_call(prompt=user_prompt, system_prompt=system_prompt, temperature=0.7)
        if not is_valid_xml(response):
            # Attempt one re-gen
            regen_prompt = (
                f"Your previous response was invalid or didn't match the {style} structure. "
                "Regenerate with the correct tags."
            )
            response = llm_call(prompt=regen_prompt, system_prompt=system_prompt, temperature=0.7)

        conversation_history.append(response)

        # For demonstration, parse each round for insights
        if style.lower() == "hegelian":
            round_data = parse_hegelian_round(response, participants)
            print(f"\n=== Hegelian Round {r+1} ===")
            for p, phases in round_data.items():
                print(f"{p} -> Thesis: {phases['Thesis']}\n"
                      f"         Antithesis: {phases['Antithesis']}\n"
                      f"         Synthesis: {phases['Synthesis']}\n")
        else:
            round_data = parse_socratic_round(response, participants)
            print(f"\n=== Socratic Round {r+1} ===")
            for p, phases in round_data.items():
                print(f"{p} -> Question: {phases['Question']}\n"
                      f"      AttemptedDefinition: {phases['AttemptedDefinition']}\n"
                      f"      CounterExample: {phases['CounterExample']}\n"
                      f"      Refinement: {phases['Refinement']}\n")

    return conversation_history


###############################################################################
# MAIN DEMO
###############################################################################

if __name__ == "__main__":
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
    # Suppose we have conversation_history from multiple calls or manual text
    conversation_history_mock = [
        "Plato introduced the idea of forms in the first round. He also mentioned the cave allegory.",
        "Nietzsche responded by challenging ideal forms, but recursion and self-reference also came up in the discussion about consciousness.",
        "They both noted emergent complexity in the interplay of philosophical concepts, referencing shadows in the cave example."
    ]

    concept_weights = advanced_track_concepts(conversation_history_mock, concept_dict)
    print("Concept Weights:", concept_weights)

    # Build a concept graph
    G = build_concept_graph(conversation_history_mock, concept_dict)
    emergent = detect_emergent_patterns(G)
    print("Emergent Patterns:", emergent)

    # 5) Dialectical Debate with Hegelian Structure
    print("\n=== HEGELIAN DIALECTICAL DEBATE DEMO ===")
    hegelian_history = run_dialectical_debate(["Aristotle", "Kant"], "metaphysics of ethics", rounds=2, style="hegelian")

    # 6) Dialectical Debate with Socratic Structure
    print("\n=== SOCRATIC DIALOGUE DEMO ===")
    socratic_history = run_dialectical_debate(["Socrates", "Descartes"], "the nature of knowledge", rounds=2, style="socratic")
