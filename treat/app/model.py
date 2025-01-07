from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime

def analyze_script(script):
    # Starting the script analysis
    print("\n=== Starting Analysis ===")
    print(f"Time: {datetime.now()}")  # Outputting the current timestamp
    print("Loading model and tokenizer...")

    try:
        # Load the tokenizer and model, selecting the appropriate device (CPU or CUDA)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", use_fast=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"  # Use CUDA if available, else use CPU
        print(f"Using device: {device}")

        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,  # Use 16-bit precision for CUDA, 32-bit for CPU
            device_map="auto"  # Automatically map model to available device
        )
        print("Model loaded successfully")

        # Define trigger categories with their descriptions
        trigger_categories = {
            "Violence": {
                "mapped_name": "Violence",
                "description": (
                    "Any act involving physical force or aggression intended to cause harm, injury, or death to a person, animal, or object. "
                    "Includes direct physical confrontations (e.g., fights, beatings, or assaults), implied violence (e.g., very graphical threats or descriptions of injuries), "
                    "or large-scale events like wars, riots, or violent protests."
                )
            },
            "Death": {
                "mapped_name": "Death References",
                "description": (
                    "Any mention, implication, or depiction of the loss of life, including direct deaths of characters, including mentions of deceased individuals, "
                    "or abstract references to mortality (e.g., 'facing the end' or 'gone forever'). This also covers depictions of funerals, mourning, "
                    "grieving, or any dialogue that centers around death, do not take metaphors into context that don't actually lead to death."
                )
            },
            "Substance Use": {
                "mapped_name": "Substance Use",
                "description": (
                    "Any explicit or implied reference to the consumption, misuse, or abuse of drugs, alcohol, or other intoxicating substances. "
                    "Includes scenes of drinking, smoking, or drug use, whether recreational or addictive. May also cover references to withdrawal symptoms, "
                    "rehabilitation, or substance-related paraphernalia (e.g., needles, bottles, pipes)."
                )
            },
            "Gore": {
                "mapped_name": "Gore",
                "description": (
                    "Extremely detailed and graphic depictions of highly severe physical injuries, mutilation, or extreme bodily harm, often accompanied by descriptions of heavy blood, exposed organs, "
                    "or dismemberment. This includes war scenes with severe casualties, horror scenarios involving grotesque creatures, or medical procedures depicted with excessive detail."
                )
            },
            "Vomit": {
                "mapped_name": "Vomit",
                "description": (
                    "Any reference to the act of vomiting, whether directly described, implied, or depicted in detail. This includes sounds or visual descriptions of the act, "
                    "mentions of nausea leading to vomiting, or its aftermath (e.g., the presence of vomit, cleaning it up, or characters reacting to it)."
                )
            },
            "Sexual Content": {
                "mapped_name": "Sexual Content",
                "description": (
                    "Any depiction or mention of sexual activity, intimacy, or sexual behavior, ranging from implied scenes to explicit descriptions. "
                    "This includes romantic encounters, physical descriptions of characters in a sexual context, sexual dialogue, or references to sexual themes (e.g., harassment, innuendos)."
                )
            },
            "Sexual Abuse": {
               "mapped_name": "Sexual Abuse",
               "description": (
                  "Any form of non-consensual sexual act, behavior, or interaction, involving coercion, manipulation, or physical force. "
                  "This includes incidents of sexual assault, molestation, exploitation, harassment, and any acts where an individual is subjected to sexual acts against their will or without their consent. "
                  "It also covers discussions or depictions of the aftermath of such abuse, such as trauma, emotional distress, legal proceedings, or therapy. "
                  "References to inappropriate sexual advances, groping, or any other form of sexual misconduct are also included, as well as the psychological and emotional impact on survivors. "
                  "Scenes where individuals are placed in sexually compromising situations, even if not directly acted upon, may also fall under this category."
                )
            },
            "Self-Harm": {
                "mapped_name": "Self-Harm",
                "description": (
                    "Any mention or depiction of behaviors where an individual intentionally causes harm to themselves. This includes cutting, burning, or other forms of physical injury, "
                    "as well as suicidal ideation, suicide attempts, or discussions of self-destructive thoughts and actions. References to scars, bruises, or other lasting signs of self-harm are also included."
                )
            },
            "Gun Use": {
                "mapped_name": "Gun Use",
                "description": (
                    "Any explicit or implied mention of firearms being handled, fired, or used in a threatening manner. This includes scenes of gun violence, references to shootings, "
                    "gun-related accidents, or the presence of firearms in a tense or dangerous context (e.g., holstered weapons during an argument)."
                )
            },
            "Animal Cruelty": {
                "mapped_name": "Animal Cruelty",
                "description": (
                    "Any act of harm, abuse, or neglect toward animals, whether intentional or accidental. This includes physical abuse (e.g., hitting, injuring, or killing animals), "
                    "mental or emotional mistreatment (e.g., starvation, isolation), and scenes where animals are subjected to pain or suffering for human entertainment or experimentation."
                )
            },
            "Mental Health Issues": {
                "mapped_name": "Mental Health Issues",
                "description": (
                    "Any reference to mental health struggles, disorders, or psychological distress. This includes mentions of depression, anxiety, PTSD, bipolar disorder, schizophrenia, "
                    "or other conditions. Scenes depicting therapy sessions, psychiatric treatment, or coping mechanisms (e.g., medication, journaling) are also included. May cover subtle hints "
                    "like a character expressing feelings of worthlessness, hopelessness, or detachment from reality."
                )
            }
        }

        print("\nProcessing text...")  # Output indicating the text is being processed
        chunk_size = 256  # Set the chunk size for text processing
        overlap = 15  # Overlap between chunks for context preservation
        script_chunks = []  # List to store script chunks

        # Split the script into smaller chunks
        for i in range(0, len(script), chunk_size - overlap):
            chunk = script[i:i + chunk_size]
            script_chunks.append(chunk)
        
        print(f"Split into {len(script_chunks)} chunks with {overlap} token overlap")  # Inform about the chunking

        identified_triggers = {}  # Dictionary to store the identified triggers

        # Process each chunk of the script
        for chunk_idx, chunk in enumerate(script_chunks, 1):
            print(f"\n--- Processing Chunk {chunk_idx}/{len(script_chunks)} ---")
            print(f"Chunk text (preview): {chunk[:50]}...")  # Preview of the current chunk
            
            # Check each category for triggers
            for category, info in trigger_categories.items():
                mapped_name = info["mapped_name"]
                description = info["description"]

                print(f"\nAnalyzing for {mapped_name}...")
                prompt = f"""
                Check this text for any indication of {mapped_name} ({description}).
                Be sensitive to subtle references or implications, make sure the text is not metaphorical.
                Respond concisely with: YES, NO, or MAYBE.
                Text: {chunk}
                Answer:
                """

                print(f"Sending prompt to model...")  # Indicate that prompt is being sent to the model
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)  # Tokenize the prompt
                inputs = {k: v.to(device) for k, v in inputs.items()}  # Send inputs to the chosen device

                with torch.no_grad():  # Disable gradient calculation for inference
                    print("Generating response...")  # Indicate that the model is generating a response
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=3,  # Limit response length
                        do_sample=True,  # Enable sampling for more diverse output
                        temperature=0.5,  # Control randomness of the output
                        top_p=0.9,  # Use nucleus sampling
                        pad_token_id=tokenizer.eos_token_id  # Pad token ID
                    )
                
                response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().upper()  # Decode and format the response
                first_word = response_text.split("\n")[-1].split()[0] if response_text else "NO"  # Get the first word of the response
                print(f"Model response for {mapped_name}: {first_word}")

                # Update identified triggers based on model response
                if first_word == "YES":
                   print(f"Detected {mapped_name} in this chunk!")  # Trigger detected
                   identified_triggers[mapped_name] = identified_triggers.get(mapped_name, 0) + 1
                elif first_word == "MAYBE":
                   print(f"Possible {mapped_name} detected, marking for further review.")  # Possible trigger detected
                   identified_triggers[mapped_name] = identified_triggers.get(mapped_name, 0) + 0.5
                else:
                   print(f"No {mapped_name} detected in this chunk.")  # No trigger detected

        print("\n=== Analysis Complete ===")  # Indicate that analysis is complete
        print("Final Results:")
        final_triggers = []  # List to store final triggers

        # Filter and output the final trigger results
        for mapped_name, count in identified_triggers.items():
            if count > 0.5:
                final_triggers.append(mapped_name)
            print(f"- {mapped_name}: found in {count} chunks")

        if not final_triggers:
            print("No triggers detected")  # No triggers detected
            final_triggers = ["None"]

        print("\nReturning results...")
        return final_triggers  # Return the list of detected triggers

    except Exception as e:
        # Handle errors and provide stack trace
        print(f"\nERROR OCCURRED: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def get_detailed_analysis(script):
    print("\n=== Starting Detailed Analysis ===")
    triggers = analyze_script(script)  # Call the analyze_script function
    
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

    print("\nFinal Result Dictionary:", result)  # Output the final result dictionary
    return result
