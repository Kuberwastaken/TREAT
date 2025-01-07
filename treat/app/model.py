from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import gc

class ContentAnalyzer:
    def __init__(self):
        self.model_name = "meta-llama/Llama-3.2-1B"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load model with memory optimization"""
        try:
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            
            print(f"Loading model on {self.device}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            return True
        except Exception as e:
            print(f"Model loading error: {str(e)}")
            return False

    def cleanup(self):
        """Clean up GPU memory"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def analyze_chunk(self, chunk, category_info):
        """Analyze a single chunk of text for a specific trigger"""
        mapped_name = category_info["mapped_name"]
        description = category_info["description"]

        prompt = f"""Check this text for any indication of {mapped_name} ({description}).
        Be sensitive to subtle references or implications, make sure the text is not metaphorical.
        Respond concisely with: YES, NO, or MAYBE.
        Text: {chunk}
        Answer:"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip().upper()
            first_word = response.split("\n")[-1].split()[0] if response else "NO"
            
            score = 1 if first_word == "YES" else 0.5 if first_word == "MAYBE" else 0
            return score, first_word
            
        except Exception as e:
            print(f"Chunk analysis error: {str(e)}")
            return 0, "NO"

    def analyze_text(self, text):
        """Main analysis function"""
        if not self.load_model():
            return {"error": "Model loading failed"}

        chunk_size = 256  # Set the chunk size for text processing
        overlap = 15  # Overlap between chunks for context preservation
        script_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

        trigger_categories = {
            "Violence": {"mapped_name": "Violence", "description": "Any act involving physical force or aggression intended to cause harm, injury, or death."},
            "Death": {"mapped_name": "Death References", "description": "Any mention, implication, or depiction of the loss of life, including direct deaths or abstract references to mortality."},
            "Substance_Use": {"mapped_name": "Substance Use", "description": "References to consumption, misuse, or abuse of drugs, alcohol, or other intoxicating substances."},
            "Gore": {"mapped_name": "Gore", "description": "Graphic depictions of severe physical injuries, mutilation, or extreme bodily harm."},
            "Sexual_Content": {"mapped_name": "Sexual Content", "description": "Depictions or mentions of sexual activity, intimacy, or sexual behavior."},
            "Self_Harm": {"mapped_name": "Self-Harm", "description": "Behaviors where an individual intentionally causes harm to themselves."},
            "Mental_Health": {"mapped_name": "Mental Health Issues", "description": "References to mental health struggles, disorders, or psychological distress."}
        }

        identified_triggers = {}

        for chunk_idx, chunk in enumerate(script_chunks, 1):
            print(f"\n--- Processing Chunk {chunk_idx}/{len(script_chunks)} ---")
            for category, info in trigger_categories.items():
                score, response = self.analyze_chunk(chunk, info)
                if response == "YES":
                    identified_triggers[category] = identified_triggers.get(category, 0) + 1
                elif response == "MAYBE":
                    identified_triggers[category] = identified_triggers.get(category, 0) + 0.5

        final_triggers = [category for category, count in identified_triggers.items() if count > 0.5]
        self.cleanup()

        if not final_triggers:
            final_triggers = ["None"]

        return final_triggers

def get_detailed_analysis(script):
    analyzer = ContentAnalyzer()
    print("\n=== Starting Detailed Analysis ===")
    triggers = analyzer.analyze_text(script)
    
    if isinstance(triggers, list) and triggers != ["None"]:
        result = {
            "detected_triggers": triggers,
            "confidence": "High - Content detected",
            "model": "Llama-3.2-1B",
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    else:
        result = {
            "detected_triggers": ["None"],
            "confidence": "High - No concerning content detected",
            "model": "Llama-3.2-1B",
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    print("\nFinal Result Dictionary:", result)
    return result