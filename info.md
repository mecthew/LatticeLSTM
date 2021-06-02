1. LatticeLSTM是直接丢弃长度超过max_len的样例
2. clip256是从FLAT获得的
3. FLAT不定义max_len，直接取FastNLP的Dataset类"seq_len"的最大值