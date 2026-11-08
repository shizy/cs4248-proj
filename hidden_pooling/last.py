from .pooler import HiddenStatePooler, register_pooler

@register_pooler("last")
class LastPooler(HiddenStatePooler):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, full_hidden_states, attention_mask):
        return full_hidden_states[-1]
