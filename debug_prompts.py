import json
from ragas.metrics import Faithfulness, ContextPrecision, AnswerCorrectness

def print_prompt(name, prompt_obj):
    print(f"\n{'='*80}")
    print(f"PROMPT: {name}")
    print(f"{'='*80}")
    if hasattr(prompt_obj, 'instruction'):
        print(f"INSTRUCTION:\n{prompt_obj.instruction}")
    if hasattr(prompt_obj, 'examples') and prompt_obj.examples:
        print(f"\nEXAMPLES ({len(prompt_obj.examples)}):")
        for i, ex in enumerate(prompt_obj.examples):
            print(f"\n--- Example {i+1} ---")
            try:
                # Try to print as dict if it's a Pydantic model
                if hasattr(ex, 'dict'):
                    print(json.dumps(ex.dict(), indent=2))
                elif hasattr(ex, 'model_dump'):
                    print(json.dumps(ex.model_dump(), indent=2))
                else:
                    print(ex)
            except Exception:
                print(ex)
    else:
        print("\nNO EXAMPLES")

f = Faithfulness()
cp = ContextPrecision()
ac = AnswerCorrectness()

def get_attr(obj, names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None

print_prompt("Faithfulness - Statement Generator", get_attr(f, ["statement_generator_prompt"]))
print_prompt("Faithfulness - NLI Statement", get_attr(f, ["n_l_i_statement_prompt", "nli_statement_prompt", "nli_statements_prompt"]))
print_prompt("Context Precision", get_attr(cp, ["context_precision_prompt"]))
print_prompt("Answer Correctness - Classifier", get_attr(ac, ["correctness_classifier"]))
print_prompt("Answer Correctness - Statement Generator", get_attr(ac, ["statement_generator_prompt"]))
