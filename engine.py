from anthropic import Anthropic
import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PhilosopherProfile:
    """Captures the essence of each philosophical voice"""
    name: str
    style: str
    key_concepts: List[str]
    era: str
    methodologies: List[str]

@dataclass
class DialoguePhase:
    """Represents a single phase of philosophical discourse"""
    type: str  # thesis, antithesis, synthesis
    content: str
    author: str
    timestamp: datetime

class PhilosophicalEngine:
    """
    Our grand machine for orchestrating philosophical dialogues across time and space!
    Like a temporal telephone exchange connecting great minds through the ages.
    """
    def __init__(self, api_key: str = None):
        self.client = Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.conversation_history: List[Dict] = []
        self.conceptual_threads: Dict[str, List[str]] = {}
        self.emergent_patterns: List[str] = []
        
    def llm_call(
        self, 
        prompt: str, 
        system_prompt: str = "", 
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_retries: int = 3
    ) -> str:
        """
        The heart of our system - a resilient interface to our AI oracle
        """
        messages = self.conversation_history + [{"role": "user", "content": prompt}]
        
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=messages,
                    temperature=temperature,
                )
                return response.content[0].text
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to get response after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
                
    def craft_dialectical_framework(
        self,
        philosophers: Dict[str, PhilosopherProfile],
        topic: str
    ) -> str:
        """
        Constructs the stage directions for our philosophical performance
        """
        xml_structure = "\n".join([
            f"<philosopher name='{p.name}'>"
            f"  <thesis>...</thesis>"
            f"  <antithesis>...</antithesis>"
            f"  <synthesis>...</synthesis>"
            f"</philosopher>"
            for p in philosophers.values()
        ])
        
        return f"""You are orchestrating a profound dialogue on {topic}.
Each philosopher must speak in their authentic voice, drawing upon:

{self._format_philosophical_backgrounds(philosophers)}

Structure each round as:
{xml_structure}

Each contribution should:
1. Reflect the philosopher's distinctive methodology
2. Build upon previous exchanges
3. Advance the dialectical progression

Use only valid XML. No content outside the specified tags."""

    def _format_philosophical_backgrounds(
        self,
        philosophers: Dict[str, PhilosopherProfile]
    ) -> str:
        """
        Creates a rich context for each philosophical voice
        """
        backgrounds = []
        for p in philosophers.values():
            background = f"""
{p.name} ({p.era}):
- Style: {p.style}
- Key concepts: {', '.join(p.key_concepts)}
- Methods: {', '.join(p.methodologies)}
"""
            backgrounds.append(background)
        return "\n".join(backgrounds)

    def validate_philosophical_discourse(
        self,
        response: str,
        philosophers: Dict[str, PhilosopherProfile]
    ) -> Tuple[bool, Optional[str]]:
        """
        Ensures both structural and philosophical integrity
        Like a quality control system for wisdom!
        """
        try:
            # First, basic XML validation
            root = ET.fromstring(f"<dialogue>{response}</dialogue>")
            
            # Track required elements for each philosopher
            philosophical_elements = {
                p.name: {
                    'thesis': False,
                    'antithesis': False,
                    'synthesis': False
                }
                for p in philosophers.values()
            }
            
            # Validate structure and track elements
            for philosopher in root.findall(".//philosopher"):
                name = philosopher.get('name')
                if name not in philosophers:
                    return False, f"Unknown philosopher: {name}"
                
                for element in philosopher:
                    if element.tag in philosophical_elements[name]:
                        philosophical_elements[name][element.tag] = True
                        
                        # Validate philosophical voice
                        if not self._validate_voice_authenticity(
                            element.text,
                            philosophers[name]
                        ):
                            return False, f"Voice mismatch for {name} in {element.tag}"
            
            # Check completeness
            for name, elements in philosophical_elements.items():
                if not all(elements.values()):
                    missing = [k for k, v in elements.items() if not v]
                    return False, f"Missing elements for {name}: {missing}"
            
            return True, None
            
        except ET.ParseError as e:
            return False, f"XML Parse Error: {str(e)}"

    def _validate_voice_authenticity(
        self,
        text: str,
        profile: PhilosopherProfile
    ) -> bool:
        """
        Ensures each philosopher maintains their unique voice
        Like a philosophical fingerprint analyzer
        """
        # Simple heuristic check for key concepts and methodologies
        text_lower = text.lower()
        concept_count = sum(1 for concept in profile.key_concepts 
                          if concept.lower() in text_lower)
        method_count = sum(1 for method in profile.methodologies 
                          if method.lower() in text_lower)
        
        # Require at least some key concepts and methodologies
        return concept_count >= 1 and method_count >= 1

    def extract_philosophical_insights(
        self,
        response: str,
        philosophers: List[str]
    ) -> Dict[str, List[DialoguePhase]]:
        """
        Harvests wisdom from our philosophical exchange
        Like mining precious gems of insight!
        """
        insights = {name: [] for name in philosophers}
        
        try:
            root = ET.fromstring(f"<dialogue>{response}</dialogue>")
            
            for philosopher in root.findall(".//philosopher"):
                name = philosopher.get('name')
                if name not in insights:
                    continue
                
                for element in philosopher:
                    if element.tag in ['thesis', 'antithesis', 'synthesis']:
                        insights[name].append(
                            DialoguePhase(
                                type=element.tag,
                                content=element.text.strip(),
                                author=name,
                                timestamp=datetime.now()
                            )
                        )
        except ET.ParseError:
            print(f"Warning: Could not parse response as XML: {response[:100]}...")
        
        return insights

    def orchestrate_dialogue(
        self,
        philosophers: Dict[str, PhilosopherProfile],
        topic: str,
        rounds: int = 3,
        temperature: float = 0.7
    ) -> Dict:
        """
        The main performance! Our philosophical orchestra in full swing!
        """
        system_prompt = self.craft_dialectical_framework(philosophers, topic)
        all_insights = []
        
        for round_num in range(rounds):
            # Compose this round's prompt
            round_prompt = self._craft_round_prompt(
                topic=topic,
                round_num=round_num,
                philosophers=philosophers
            )
            
            # Get response from our AI oracle
            response = self.llm_call(
                prompt=round_prompt,
                system_prompt=system_prompt,
                temperature=temperature
            )
            
            # Validate the response
            is_valid, error_msg = self.validate_philosophical_discourse(
                response,
                philosophers
            )
            
            if not is_valid:
                # Request clarification if needed
                response = self._request_clarification(
                    original_response=response,
                    error_msg=error_msg,
                    system_prompt=system_prompt,
                    philosophers=philosophers
                )
            
            # Extract and store insights
            round_insights = self.extract_philosophical_insights(
                response,
                list(philosophers.keys())
            )
            all_insights.append(round_insights)
            
            # Update conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            
            # Update conceptual threads
            self._update_conceptual_threads(round_insights)
            
            # Look for emergent patterns
            self._identify_emergent_patterns(round_insights)
            
            print(f"\n=== Round {round_num + 1} Complete ===")
            self._display_round_results(round_insights)
        
        return {
            'insights': all_insights,
            'threads': self.conceptual_threads,
            'patterns': self.emergent_patterns
        }

    def _craft_round_prompt(
        self,
        topic: str,
        round_num: int,
        philosophers: Dict[str, PhilosopherProfile]
    ) -> str:
        """
        Crafts the perfect invitation for each round of dialogue
        """
        base_prompt = f"""Round {round_num + 1} on {topic}.
Each philosopher should advance their position while engaging with previous arguments."""

        if self.conceptual_threads:
            thread_summary = "\nKey threads so far:\n" + "\n".join(
                f"- {thread}: {', '.join(points[:3])}"
                for thread, points in self.conceptual_threads.items()
            )
            base_prompt += thread_summary

        if self.emergent_patterns:
            pattern_summary = "\nEmerging patterns:\n" + "\n".join(
                f"- {pattern}" for pattern in self.emergent_patterns[-3:]
            )
            base_prompt += pattern_summary

        return base_prompt

    def _request_clarification(
        self,
        original_response: str,
        error_msg: str,
        system_prompt: str,
        philosophers: Dict[str, PhilosopherProfile]
    ) -> str:
        """
        Gracefully requests a more structured response
        Like a philosophical copy editor!
        """
        clarification_prompt = f"""
Your previous response requires refinement: {error_msg}

Please provide a new response that:
1. Uses valid XML structure
2. Includes all required elements (thesis, antithesis, synthesis)
3. Maintains each philosopher's authentic voice

Previous attempt:
{original_response}
"""
        return self.llm_call(
            prompt=clarification_prompt,
            system_prompt=system_prompt,
            temperature=0.5  # Lower temperature for more structured output
        )

    def _update_conceptual_threads(
        self,
        round_insights: Dict[str, List[DialoguePhase]]
    ) -> None:
        """
        Tracks the evolution of key ideas through the dialogue
        Like a philosophical genealogist!
        """
        for philosopher_insights in round_insights.values():
            for phase in philosopher_insights:
                # Extract key concepts using simple keyword matching
                # In a real implementation, you might use more sophisticated NLP
                words = phase.content.lower().split()
                for word in words:
                    if len(word) > 5:  # Simple filter for significant words
                        self.conceptual_threads.setdefault(word, []).append(
                            f"{phase.author} ({phase.type}): {phase.content[:100]}..."
                        )

    def _identify_emergent_patterns(
        self,
        round_insights: Dict[str, List[DialoguePhase]]
    ) -> None:
        """
        Discovers unexpected connections and patterns
        Like a philosophical constellation finder!
        """
        # Look for similar concepts across different philosophers
        all_phrases = []
        for philosopher, insights in round_insights.items():
            for phase in insights:
                all_phrases.append((philosopher, phase.type, phase.content))
        
        # Simple pattern detection based on shared keywords
        # In a real implementation, you might use more sophisticated similarity metrics
        for i, (phil1, type1, content1) in enumerate(all_phrases):
            for phil2, type2, content2 in all_phrases[i+1:]:
                if phil1 != phil2:
                    shared_words = set(content1.lower().split()) & set(content2.lower().split())
                    if len(shared_words) > 5:  # Arbitrary threshold
                        pattern = f"Connection found: {phil1} and {phil2} both discuss {', '.join(list(shared_words)[:3])}"
                        if pattern not in self.emergent_patterns:
                            self.emergent_patterns.append(pattern)

    def _display_round_results(
        self,
        round_insights: Dict[str, List[DialoguePhase]]
    ) -> None:
        """
        Presents the fruits of our philosophical labor
        Like a museum curator of wisdom!
        """
        for philosopher, insights in round_insights.items():
            print(f"\n{philosopher}'s Contribution:")
            for phase in insights:
                print(f"\n{phase.type.upper()}:")
                print(phase.content)
                print("-" * 40)

# Example usage
if __name__ == "__main__":
    # Define our philosophical cast
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

    # Create our philosophical engine
    engine = PhilosophicalEngine()
    
    # Let the dialogue begin!
    results = engine.orchestrate_dialogue(
        philosophers=philosophers,
        topic="the nature of consciousness and self-reference",
        rounds=3,
        temperature=0.7
    )
    
    print("\n=== Final Emergent Patterns ===")
    for pattern in results['patterns']:
        print(f"- {pattern}")
