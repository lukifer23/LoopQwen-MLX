import math
import mlx.core as mx

# Fixed standalone documents, authored for this project and excluded from training.
PROSE = [
    "The rain stopped just before dawn. Water dripped from the branches into the stream below.",
    "As the wheel turns, each spoke passes the same point once per revolution.",
    "The library closed early on Friday because the heating system needed repairs.",
    "A sample of river water can contain sediment, dissolved minerals, and living organisms.",
    "She folded the map along the old crease and put it back in her pocket.",
    "To measure a shadow accurately, mark both ends before the light changes.",
]


def language_loss(model, tokenizer):
    total = tokens = 0
    for text in PROSE:
        x = mx.array([tokenizer.encode(text, add_special_tokens=False)])
        total += mx.fast.cross_entropy(model(x[:,:-1]), x[:,1:]).sum().item()
        tokens += x.size-1
    return dict(nll=total/tokens, perplexity=math.exp(total/tokens), tokens=tokens)
