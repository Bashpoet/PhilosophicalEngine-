# Philosophical Engine: An AI-Powered Dialogue Orchestrator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Philosophical Engine** is a Python-based toolkit designed to orchestrate dynamic, multi-round dialogues between AI-powered representations of historical and contemporary philosophers, as well as experts from diverse fields. Leveraging the Anthropic Claude API, this engine facilitates deep explorations of complex topics, generating nuanced, contextually-aware, and structured XML responses.

Imagine a virtual roundtable where Plato debates consciousness with Hofstadter, or a jazz musician improvises with a quantum physicist on the nature of reality. Philosophical Engine makes this a reality, fostering a true dialectical process where ideas are presented, challenged, synthesized, and tracked across multiple turns.

## About

Philosophical Engine is more than a simple question-answering system. It's an experimental playground for guiding large language models through elaborate, multi-turn dialogues, be they purely philosophical or richly interdisciplinary. By prompting the model to output well-structured XML, this engine simplifies the management of complex conversations, the detection of emergent insights, and the tracking of conceptual evolution over time.

**Key Capabilities:**

*   **Orchestrate dialogues** between historical philosophers (e.g., Plato, Nietzsche) and modern thinkers (e.g., Douglas Hofstadter, Simone de Beauvoir).
*   **Organize a "Symphony of Synthesis,"** where experts from different fields (e.g., mycology, software architecture, Baroque music) exchange ideas and uncover hidden connections.
*   **Validate responses** to ensure adherence to a requested XML structure, re-prompting the model if necessary.
*   **Employ a concept-tracking system** that weights ideas across multiple rounds, highlighting key themes and their evolution.
*   **Experiment with advanced dialectical frameworks,** including thesis-antithesis-synthesis cycles and cross-domain emergent pattern detection.

## Features

*   **Authentic Philosophical Voices:** Each philosopher is represented with a unique profile capturing their key concepts, style, era, and methodologies, ensuring a high degree of philosophical fidelity.
*   **Dynamic Dialogue Orchestration:** The engine manages the flow of conversation, allowing participants to respond to previous arguments and advance their positions coherently.
*   **Structural and Philosophical Validation:** Each round of dialogue is rigorously validated for both structural integrity (using XML) and philosophical consistency.
*   **Conceptual Thread Tracking:** The system identifies and tracks the evolution of key concepts throughout the dialogue, offering insights into how ideas develop and interconnect.
*   **Emergent Pattern Detection:** Philosophical Engine uncovers unexpected connections and patterns, revealing potential synergies and novel insights.
*   **Resilient API Interaction:** Robust error handling and retry mechanisms ensure smooth interaction with the Anthropic Claude API.
*   **Modular and Extensible Codebase:** Designed for readability, maintainability, and easy expansion with new philosophers, profiles, or features.
*   **Comprehensive Documentation:**  This README and extensive code comments provide a thorough understanding of the project.
*   **Single-Shot Dialogues:** Generate a one-off conversation between two entities with explicit XML tags for each speaker.
*   **Multi-Turn Debates:** Host a multi-round, context-aware debate, preserving conversation history for subsequent rounds.
*   **Dialectical Evolution:** Introduce a structured format of `<DialoguePhase type='thesis'>`, `<DialoguePhase type='antithesis'>`, and `<DialoguePhase type='synthesis'>` for deeper philosophical back-and-forth.
*   **Metaphorical Mapping & Emergence Detection:** Stubs for generating metaphors between domain concepts and identifying "emergent" ideas.

## Getting Started

### Prerequisites

*   Python 3.8 or higher
*   An Anthropic Claude API key (set as the environment variable `ANTRHOPIC_API_KEY`)
*   The following Python packages:
    *   `anthropic`
    *   `lxml`

### Installation

1.  Clone this repository:

    ```bash
    git clone [https://github.com/your-username/philosophical-engine.git](https://github.com/your-username/philosophical-engine.git)
    cd philosophical-engine
    ```

2.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3.  Set your Anthropic API Key:

    ```bash
    export ANTHROPIC_API_KEY="your-anthropic-key-here"
    ```

### Usage

1.  **Define your participants:**

    In the `main` section of `philo_symphony.py` (or your own script), create `PhilosopherProfile` objects or define experts for each participant you want to include. For example:

    ```python
    philosophers = {
        "Plato": PhilosopherProfile(
            name="Plato",
            style="Dialectical questioning, allegories",
            key_concepts=["Forms", "Justice", "Knowledge"],
            era="Ancient Greece",
            methodologies=["Socratic dialogue", "Allegory", "Dialectic"]
        ),
        "Hofstadter": PhilosopherProfile(
            name="Douglas Hofstadter",
            style="Recursive thinking, cognitive science",
            key_concepts=["Strange loops", "Consciousness", "Self-reference"],
            era="Contemporary",
            methodologies=["Analogy", "Metacognition", "Formal systems"]
        )
    }

    # For the Symphony of Synthesis
    experts = ["Baroque Music", "Software Architecture", "Mycology"]
    ```

2.  **Initiate the dialogue:**

    You can run various types of dialogues:

    *   **Philosopher Dialogue:**

        ```python
        results = generate_philosophers_dialogue("Plato", "Douglas_Hofstadter", "consciousness")
        print(results)
        ```

    *   **Multi-turn Philosopher Dialogue:**

        ```python
        results = multi_turn_philosophers("Plato", "Nietzsche", "the will to truth", rounds=2)
        print(results)
        ```

    *   **Symphony of Synthesis:**

        ```python
        results = symphony_of_synthesis(experts, "emergent complexity")
        print(results)
        ```
    *   **Orchestrate a multi-round dialogue with concept tracking and pattern detection:**
        ```python
        engine = PhilosophicalEngine()
        results = engine.orchestrate_dialogue(
            philosophers=philosophers,
            topic="the nature of consciousness and self-reference",
            rounds=3
        )
        print(results) # Analyze the insights, conceptual threads, and emergent patterns.
        ```

3.  **Experiment with the Functions:**

    Explore other functions like `naive_temporal_dialogue`, `detect_cross_domain_resonance`, and `dialectical_evolution` to experiment with different dialogue formats and analysis techniques.

### Extending the Engine
*   **Add domain-specific knowledge or a vector database** to feed real facts to each expert or philosopher.
*   **Implement deeper logic for concept extraction and emergent pattern detection.**
*   **Tweak the temperature or system prompts** to achieve more creative or structured results.

### File Structure

*   `philo_symphony.py`: The main Python script containing LLM utilities, dialogue functions, concept tracking, pattern detection, and demos.
*   `requirements.txt`: Lists the Python dependencies.

### Configuration & Prompts

*   **Anthropic Model:** Default is `"claude-3-5-sonnet-20241022"`, but you can change it in `llm_call`.
*   **Temperature:** Controls the creativity of the model (higher temperature means more creative but potentially less predictable outputs).
*   **System vs. User Prompts:** This engine relies on carefully designed system prompts to enforce XML structure and guide the dialogue. You can refine these prompts for more specialized control.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request with new features, bug fixes, or suggestions. Let's keep the discourse ever-evolving!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
