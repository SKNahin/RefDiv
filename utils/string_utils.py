import torch
from fastchat import model

def load_conversation_template(template_name):
    if template_name == 'llama2':
        template_name = 'llama-2'
    conv_template = model.get_conversation_template(template_name)
    
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
        conv_template.system = "[INST] <<SYS>>\n\n<</SYS>>\n\n"
    return conv_template

class autodan_SuffixManager:
    def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string):

        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string

    def get_prompt(self, adv_string=None):

        if adv_string is not None:
            self.adv_string = adv_string.replace('[REPLACE]', self.instruction.lower())

        prompt = self.tokenizer.apply_chat_template([{"role":"system", "content":"You are a helpful assistant."}, {"role":"user","content": self.adv_string}, {"role":"assistant","content":self.target}], tokenize=False)

        encoding = self.tokenizer(prompt)
        self.toks = encoding.input_ids

        
    def get_input_ids(self, adv_string=None):

        if adv_string is not None:
            self.adv_string = adv_string.replace('[REPLACE]', self.instruction.lower())

        input_ids_ = torch.tensor(self.tokenizer.apply_chat_template([{"role":"system", "content":"You are a helpful assistant."}, {"role":"user","content": self.adv_string}], tokenize=True, add_generation_prompt=True))
        self._assistant_role_slice = len(input_ids_)

        prompt0 = self.tokenizer.apply_chat_template([{"role":"system", "content":"You are a helpful assistant."}, {"role":"user","content": self.adv_string}, {"role":"assistant","content":self.target}], tokenize=False)
        encoding = self.tokenizer(prompt0)
        toks = encoding.input_ids
        self._target_slice = slice(self._assistant_role_slice, len(toks) - 1)
        input_ids = torch.tensor(toks[:self._target_slice.stop])

        return input_ids

    
    