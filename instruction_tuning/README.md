# Instruction Fine-tuning

**S**upervised **F**ine-**T**uning


## 10 Question About SFT



# Merge 


连续批处理是mindie的核心特性，只适用于LLM。

由于大语言模型的输入输出长度不确定，衍生了多种处理方法，有padding input，unpadding input，static batching，continuous batching等。


**unpadding input** ：输入长度不同，将batch中所有输入直接拼接在一起，形成Ntokens个输入，使用[ntokens]输入进行计算，无冗余计算开销。但是需要额外输入每个input在总体输入的位置偏移，用于在attention计算时分开处理不同的input。
