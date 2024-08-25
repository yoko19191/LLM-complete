# -*- coding: utf-8 -*-

# @File    : BaichuanModel.py
# @Date    : 2024-04-12
# @Author  : jiang
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel


class BaichuanModel(object):
    def __init__(self):
        self.max_new_tokens = 2048  # 每轮对话最多生成多少个token
        self.history_max_len = 2048  # 模型记忆的最大token长度
        self.top_p = 0.6
        self.temperature = 0.9
        self.repetition_penalty = 1.1
        self.device = 'cuda'
        self.model_name_or_path = "/root/autodl-tmp/qwen/Qwen15-14B-Chat"
        self.adapter_name_or_path = "/root/autodl-tmp/checkpoints/qwen/zhihu_pastime_mind_role_merge_common/lora/checkpoint-4500"
        self.init_model()

    def init_model(self, load_in_4bit=True):
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        else:
            quantization_config = None

        # 加载base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            load_in_4bit=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map='auto',
            quantization_config=quantization_config
        )
        # 加载adapter
        if self.adapter_name_or_path is not None:
            self.model = PeftModel.from_pretrained(self.model, self.adapter_name_or_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path,
                                                       trust_remote_code=True,
                                                       use_fast=False)

    def pack_messages(self, input_history, need_history):
        messages = []
        if need_history:
            for item in input_history:
                question = item['question']
                response = item['response']
                messages.append([question, response])
        return messages

    def chat(self, user_input, input_history):
        user_input = user_input.strip()
        user_input = '<reserved_106>' + user_input + '<reserved_107>'
        model_input_ids = self.tokenizer_history(input_history, user_input)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=model_input_ids, max_new_tokens=self.max_new_tokens, do_sample=True, top_p=self.top_p,
                temperature=self.temperature, repetition_penalty=self.repetition_penalty
            )

        model_input_ids_len = model_input_ids.size(1)
        response_ids = outputs[:, model_input_ids_len:]
        response = self.tokenizer.batch_decode(response_ids)
        answer = response[0].strip().replace(self.tokenizer.eos_token, "")
        return answer

    def chatSystemExample(self, user_input, input_history, system):
        user_input = (system + user_input).strip()
        user_input = '<reserved_106>' + user_input + '<reserved_107>'
        model_input_ids = self.tokenizer_system(user_input, system)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=model_input_ids, max_new_tokens=self.max_new_tokens, do_sample=True, top_p=self.top_p,
                temperature=self.temperature, repetition_penalty=self.repetition_penalty
            )

        model_input_ids_len = model_input_ids.size(1)
        response_ids = outputs[:, model_input_ids_len:]
        response = self.tokenizer.batch_decode(response_ids)
        answer = response[0].strip().replace(self.tokenizer.eos_token, "")
        return answer

    def tokenizer_history(self, input_history, user_input):
        history = self.pack_messages(input_history, need_history=True)
        user_input_ids = self.tokenizer(user_input, return_tensors="pt").input_ids
        history_token_ids = torch.tensor([[]], dtype=torch.long)

        #         _register_template(
        #     name="qwen",
        #     format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
        #     format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
        #     format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
        #     format_separator=EmptyFormatter(slots=["\n"]),
        #     default_system="You are a helpful assistant.",
        #     stop_words=["<|im_end|>"],
        #     replace_eos=True,
        # )

        #         _register_template(
        #     name="baichuan2",
        #     format_user=StringFormatter(slots=["<reserved_106>{{content}}<reserved_107>"]),
        #     efficient_eos=True,
        # )

        # 拼接聊天历史
        for content in history:
            u1 = content[0]
            u2 = content[1]
            u1 = '<|im_start|>' + u1.strip() + '<|im_end|>'
            u1_ids = self.tokenizer(u1, return_tensors="pt").input_ids
            u2_ids = self.tokenizer.encode_plus(u2, return_tensors="pt")["input_ids"]
            u2_id_list = u2_ids.tolist()[0]
            u2_id_list = u2_id_list + [66, 2]
            u2_ids = torch.tensor([u2_id_list])
            history_token_ids = torch.concat((history_token_ids, u1_ids), dim=1)
            history_token_ids = torch.concat((history_token_ids, u2_ids), dim=1)

        # 加入输入问题
        history_token_ids = torch.concat((history_token_ids, user_input_ids), dim=1)
        # 截断
        model_input_ids = history_token_ids.to(self.device)
        return model_input_ids
        
    def tokenizer_system(self, user_input, system):
        user_input_ids = self.tokenizer(user_input, return_tensors="pt").input_ids
        history_token_ids = torch.tensor([], dtype=torch.long)
        history_token_ids = torch.concat((history_token_ids, user_input_ids), dim=1)
        # 截断
        model_input_ids = history_token_ids.to(self.device)
        return model_input_ids


baichuanModel = BaichuanModel()

while True:
    baichuanModel.chat()