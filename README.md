```markdown
# Philosophical Engine: An AI-Powered Multi-Turn Dialogue Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Philosophical Engine** is a Python-based toolkit that orchestrates dynamic, multi-round dialogues between AI-powered representations of both historical and contemporary philosophers, as well as experts from diverse fields. By leveraging the Anthropic Claude API, it enables deep explorations of complex topics, generating nuanced, contextually-aware, and structured XML responses.

Imagine staging a roundtable where Plato debates consciousness with Hofstadter, or a jazz musician improvises with a quantum physicist about the nature of reality. Philosophical Engine makes these conversations a reality, nurturing a robust dialectical process in which ideas are presented, challenged, synthesized, and tracked across multiple turns.

---

## About

Philosophical Engine transcends basic question-answering, aiming to guide large language models through elaborate, multi-turn dialogues—spanning purely philosophical debates to richly interdisciplinary explorations. By enforcing a well-structured XML output, it simplifies managing complex conversations, detecting emergent insights, and tracking how concepts evolve over time.

**Key Capabilities:**

- **Orchestrate dialogues** between historical philosophers (e.g. Plato, Nietzsche) and modern thinkers (e.g. Douglas Hofstadter, Simone de Beauvoir).  
- **Coordinate a "Symphony of Synthesis,"** inviting experts across domains (e.g. mycology, software architecture, baroque music) to exchange ideas and uncover hidden synergies.  
- **Validate responses** for well-formed XML structure, re-prompting the model if necessary.  
- **Leverage a concept-tracking system** to weight ideas across multiple rounds and highlight key thematic developments.  
- **Experiment with dialectical frameworks** like thesis–antithesis–synthesis or cross-domain emergent pattern detection.

---

## Features

- **Authentic Philosophical Voices:** Each philosopher or expert can be tailored with distinctive profiles, capturing style, era, and methodologies.  
- **Dynamic Dialogue Orchestration:** The engine manages multi-round conversational flow, letting participants reference and refine earlier arguments.  
- **Structural & Philosophical Validation:** Each round is validated for both structural (XML) correctness and internal philosophical consistency.  
- **Conceptual Thread Tracking:** Identifies how key concepts surface and intertwine across the discussion.  
- **Emergent Pattern Detection:** Highlights unexpected connections and synergies that arise in the evolving conversation.  
- **Resilient API Interaction:** Includes retry logic and error handling for smooth, robust communication with the Anthropic Claude API.  
- **Modular, Extensible Codebase:** Well-commented, easily adaptable to new personas, features, or specialized use cases.  
- **Single-Shot Dialogues:** Launches a standalone conversation with two distinct voices labeled by XML tags.  
- **Multi-Turn Debates:** Preserves context across multiple rounds for deeper, iterative explorations.  
- **Dialectical Evolution:** Supports `<DialoguePhase type='thesis'>`, `<DialoguePhase type='antithesis'>`, and `<DialoguePhase type='synthesis'>` to capture rich argumentative structure.  
- **Metaphorical Mapping & Emergence Detection:** Includes stubs for mapping domain concepts and spotting novel, “emergent” ideas or patterns.

---

## Getting Started

### Prerequisites

- Python **3.8** or higher  
- An Anthropic Claude API key (exported as `ANTHROPIC_API_KEY` in your environment)  
- The following Python packages:
  - **anthropic**
  - **scikit-learn**
  - **networkx**
  - (And other libraries listed in `requirements.txt`)

### Installation

1. **Clone this repository**:

   ```bash
   git clone https://github.com/your-username/philosophical-engine.git
   cd philosophical-engine
   ```

2. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set your Anthropic API Key**:

   ```bash
   export ANTHROPIC_API_KEY="your-anthropic-key-here"
   ```

---

## Usage

1. **Define Philosophers or Experts:**

   In your primary script (e.g. `philo_symphony_extended.py`), optionally create data structures or dictionaries for participants. Example:

   ```python
   philosophers = {
       "Plato": {
           "style": "Dialectical questioning, allegories",
           "key_concepts": ["Forms", "Justice", "Knowledge"],
           "era": "Ancient Greece"
       },
       "Hofstadter": {
           "style": "Recursive thinking, cognitive science",
           "key_concepts": ["Strange loops", "Consciousness", "Self-reference"],
           "era": "Contemporary"
       }
   }
   ```

2. **Initiate the Dialogue:**

   - **Single-Shot Philosopher Dialogue**:
     ```python
     from philo_symphony_extended import generate_philosophers_dialogue

     generate_philosophers_dialogue("Plato", "Douglas_Hofstadter", "consciousness")
     ```
   - **Multi-Turn Debate**:
     ```python
     from philo_symphony_extended import multi_turn_philosophers

     multi_turn_philosophers("Plato", "Nietzsche", "the will to truth", rounds=2)
     ```
   - **Symphony of Synthesis**:
     ```python
     from philo_symphony_extended import symphony_of_synthesis

     experts = ["Baroque Music", "Software Architecture", "Mycology"]
     symphony_of_synthesis(experts, "emergent complexity")
     ```
   - **Dialectical Debate (Hegelian or Socratic)**:
     ```python
     from philo_symphony_extended import run_dialectical_debate

     run_dialectical_debate(["Aristotle", "Kant"], "metaphysics of ethics", rounds=2, style="hegelian")
     run_dialectical_debate(["Socrates", "Descartes"], "the nature of knowledge", rounds=2, style="socratic")
     ```

3. **Experiment with Concept Tracking & Emergence**:

   ```python
   from philo_symphony_extended import advanced_track_concepts, build_concept_graph, detect_emergent_patterns

   conversation_history = [
       "Plato introduced the concept of forms and the cave allegory.",
       "Nietzsche challenged ideal forms, referencing recursion."
   ]

   concept_dict = {
       "Forms": ["forms", "platonic", "ideal"],
       "Recursion": ["recursion", "loop", "self-reference"],
   }

   weights = advanced_track_concepts(conversation_history, concept_dict)
   print("Concept Weights:", weights)

   G = build_concept_graph(conversation_history, concept_dict)
   emergent_insights = detect_emergent_patterns(G)
   print("Emergent Patterns:", emergent_insights)
   ```

---

## Extending the Engine

- **Integrate domain-specific knowledge** or use a vector database for factual context.  
- **Refine concept extraction** for more accurate or semantically rich tracking.  
- **Customize prompts and temperatures** for different levels of creativity or determinism.  
- **Develop specialized dialectical structures**—like more advanced Socratic frameworks or additional phases.  

---

## File Structure

- `philo_symphony_extended.py`  
  A single Python file demonstrating LLM utilities, dialogue orchestration, concept tracking, pattern detection, and multiple dialectical styles.

- `requirements.txt`  
  Lists Python dependencies.

---

## Configuration & Prompts

- **Anthropic Model**  
  Uses `"claude-3-5-sonnet-20241022"` by default. Change it in `llm_call` if you prefer a newer or custom Claude variant.

- **Temperature**  
  Higher values (e.g. `0.7+`) yield more creative outputs, while lower values (e.g. `0.1`) ensure more deterministic responses.

- **System vs. User Prompts**  
  The engine relies on carefully crafted system prompts to enforce XML structure and guide the model. Refine these prompts for more specialized control or stricter format requirements.

---

## Contributing

Contributions are welcome! If you have new features, bug fixes, or suggestions, please open an issue or create a pull request. Let’s continue evolving this engine to push the boundaries of AI-assisted philosophical and interdisciplinary dialogues.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
```

