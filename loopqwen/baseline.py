from .evaluation.generation_eval import generate
from .evaluation.loss_eval import language_loss
from .model_loader import load_base
from .io import save


def baseline():
    model, tokenizer, info = load_base()
    prompt = "The capital of France is"
    generate(model, tokenizer, prompt, 2)
    result = dict(
        model=info,
        generation=generate(model, tokenizer, prompt),
        prose=language_loss(model, tokenizer),
    )
    save("outputs/baseline.json", result)
    return result
