try:
    from transformer_engine.pytorch import LayerNormLinear, Linear, LayerNorm
    from transformer_engine.common.recipe import Format, DelayedScaling
except:
    print("WARNING: transformer_engine not installed. Using default recipe.")

