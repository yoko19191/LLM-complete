from torch.utils.data import Dataset
import datasets
import random
from utils import *
from transformers import Trainer
import logging
import transformers
import os
import torch
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    template_variation: bool = field(
        default=False, metadata={"help": "whether to use template variation"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

PROMPT_TEMPLATE = {
'prompt_with_input': 'The instruction is a description of the task. You need to follow that and respond to the paired input.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:',
'prompt_without_input': 'The instruction is a description of the task. You need to follow that and respond.\n\n### Instruction:\n{instruction}\n\n### Response:'
}

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        

class SFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SFTDataset, self).__init__()
        logging.warning("加载数据ing")
        
        if os.path.exists(data_path):
            train_data = datasets.load_dataset('json', data_files={'train': data_path})
        else:
            train_data = datasets.load_dataset(data_path)
            
        logging.warning('使用template将数据组合成question, answer pair')

        questions = []

        prompt_with_input, prompt_without_input = PROMPT_TEMPLATE['prompt_with_input'], PROMPT_TEMPLATE['prompt_without_input']
        for item in train_data['train']:
            if item.get("input", "") != "":
                questions.append(prompt_with_input.format_map(item))
            else:
                questions.append(prompt_without_input.format_map(item))   
                
        responses = [f"{item['output']}{tokenizer.eos_token}" for item in train_data['train']]
        
        self.questions = questions
        self.responses = responses
        
    def __len__(self):
        return len(self.questions)

    def __getitem__(self, i):
        return dict(input_ids=self.questions[i], labels=self.responses[i])    
    
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )
    
def preprocess(
    questions: Sequence[str],
    responses: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [q + r for q, r in zip(questions, responses)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, questions)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class SFTDataCollator(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        questions = []
        responses = []
        for instance in instances:
            question = instance['input_ids']
            response = instance['labels']
            questions.append(question)
            responses.append(response)

        data_dict = preprocess(questions, responses, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        
def sft_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    
    train_dataset = SFTDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = SFTDataCollator(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    logging.warning('加载模型')
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map = 'auto'
    )
    
    logging.warning('加载tokenizer, 并为tokenizer添加特殊token, 并且为新添加的token增加embedding')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
        
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    data_module = sft_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    
    
if __name__=='__main__':
    train()