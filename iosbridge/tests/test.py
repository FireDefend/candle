from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("/Users/xigsun/Documents/repo/candle/candle-examples/examples/marian-mt/opus-mt-zh-en")

model = AutoModelForSeq2SeqLM.from_pretrained("/Users/xigsun/Documents/repo/candle/candle-examples/examples/marian-mt/opus-mt-zh-en")
# 准备要翻译的文本
chinese_text = "求真务实是中国共产党人的重要思想和工作方法。前不久举行的中央经济工作会议上，习近平总书记着眼于做好明年经济工作、巩固和增强经济回升向好态势，对抓落实提出了明确要求，强调“要求真务实抓落实”“坚决纠治形式主义、官僚主义”。"  # 这里你可以替换成任何你想要翻译的中文文本
chinese_text = "在上一篇文章中我们介绍了注意力机制—目前在深度学习中被广泛应用。注意力机制能够显著提高神经机器翻译任务的性能。本文将会看一看Transformer---加速训练注意力模型的方法。Transformers在很多特定任务上已经优于Google神经机器翻译模型了。不过其最大的优点在于它的并行化训练。Google云强烈建议使用TPU云提供的Transformer模型。我们赶紧撸起袖子拆开模型看一看内部究竟如何吧。"  # 这里你可以替换成任何你想要翻译的中文文本

# 对文本进行编码
encoded_text = tokenizer.encode(chinese_text, return_tensors="pt")
tokenizer.decode(encoded_text[0], skip_special_tokens=True)


inputs = tokenizer(chinese_text, return_tensors="pt")
eos_token_id = model.config.eos_token_id
decoder_input_ids = torch.tensor([[eos_token_id]], dtype=torch.long) 
encoder_outputs = model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True).encoder_last_hidden_state

decoder_start_token = model.config.decoder_start_token_id
decoder_input_ids = torch.full((inputs.input_ids.shape[0], 1), decoder_start_token, dtype=torch.long)
outputs = model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
outputs.logits
import torch.nn.functional as F
probabilities = F.softmax(outputs.logits, dim=-1)
next_word_logits = probabilities[:, -1, :]
next_word = torch.argmax(next_word_logits, dim=-1)
# 进行翻译
translated_tokens = model.generate(encoded_text)

# 解码翻译结果
translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

print(translated_text)
import torch.nn.functional as F
import torch

# 保存模型权重
torch.save(model.state_dict(), '/Users/xigsun/Documents/repo/candle/candle-examples/examples/marian-mt/opus-mt-zh-en/model.pth')
content = torch.load('/Users/xigsun/Documents/repo/candle/candle-examples/examples/marian-mt/opus-mt-zh-en/model.pth')
model.load_state_dict(torch.load('/Users/xigsun/Documents/repo/candle/candle-examples/examples/marian-mt/opus-mt-zh-en/model.pth'))

model.save_pretrained("/Users/xigsun/Documents/repo/candle/candle-examples/examples/marian-mt/opus-mt-zh-en/model111", save_format="safetensors")


beam_width = 6
max_length = 1000

import time

# 记录开始时间
start_time = time.time()
for n in range(1000):
    seq = torch.tensor([[65000,    21802+n,  n+100]])
    outputs = model(**inputs, decoder_input_ids=seq, return_dict=True)
    logits = outputs.logits[:, 2, :].unsqueeze(1)

end_time = time.time()

# 计算并打印执行时间
elapsed_time = end_time - start_time
print(f"The code took {elapsed_time} seconds to execute.")

start_time = time.time()
best_seq = beam_search(model, inputs, decoder_input_ids, beam_width, max_length)
end_time = time.time()
# 计算并打印执行时间
elapsed_time = end_time - start_time
print(f"The code took {elapsed_time} seconds to execute.")

translated_text = tokenizer.decode(best_seq[0], skip_special_tokens=True)

def beam_search(model, input_ids, decoder_input_ids, beam_width, max_length):
    # 初始化束
    beams = [(decoder_input_ids, 0)]  # (序列, 累积得分)
    for length in range(max_length):
        new_beams = []
        for beam in beams:
            # 获取当前序列和得分
            seq, score = beam
            # 检查序列是否已经完成（例如，检查是否包含 EOS）
            if seq[0][seq.shape[1]-1] == 0:
                new_beams.append(beam)
                continue
            # 获得模型预测          
            outputs = model(**input_ids, decoder_input_ids=seq, return_dict=True)
            logits = outputs.logits[:, length, :].unsqueeze(1)
            probabilities = F.softmax(logits, dim=-1)
            # 获取 top K 个候选词汇
            topk_probs, topk_indices = probabilities.topk(beam_width)
            # 为每个候选词汇创建新的束
            for prob, idx in zip(topk_probs[0][0], topk_indices[0][0]):
                new_seq = torch.cat([seq[0], idx.unsqueeze(0)], dim=0).unsqueeze(0)
                new_score = score + torch.log(prob)
                new_beams.append((new_seq, new_score))
        # 保留总分最高的 top K 个束
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        # print(beams[0])
        print(f"The code {length} current time {time.time()} seconds to execute.")
    #print(beams)
    # 选择得分最高的序列
    best_seq, best_score = max(beams, key=lambda x: x[1])
    return best_seq


for token_id in best_seq[0]:
    # 解码单个token
    token = tokenizer.decode([token_id])
    print(f"Token ID: {token_id}, Token: {token}")

import json

# 打开JSON文件
with open('/Users/xigsun/Documents/repo/candle/candle-examples/examples/marian-mt/opus-mt-zh-en/tokenizer-marian-base-en.json', 'r') as file:
    # 解析JSON数据
    data1 = json.load(file)
data1.get("model")["vocab"][0]
data1.get("model")["vocab"][3][0]
# 现在 data 是一个Python字典，包含了JSON文件中的数据

with open('/Users/xigsun/Documents/repo/candle/candle-examples/examples/marian-mt/opus-mt-zh-en/tokenizer-marian-base-zh.json', 'r') as file:
    # 解析JSON数据
    datazh = json.load(file)
datazh.get("model")["vocab"][0]
datazh.get("model")["vocab"][3][0]
range(6)
count = 0

for n in range(65000):
  if(datazh.get("model")["vocab"][n][0] == '<NIL>' and data1.get("model")["vocab"][n][0] != '<NIL>'):
      datazh.get("model")["vocab"][n] = data1.get("model")["vocab"][n]
      count = count + 1

# Save 'datazh' to a file
with open('/Users/xigsun/Documents/repo/candle/candle-examples/examples/marian-mt/opus-mt-zh-en/tokenizer-marian-base-big.json', 'w', encoding='utf-8') as file:
    json.dump(datazh, file, ensure_ascii=False, indent=4)


with open(tokenizer.spm_files[0], "rb") as f:
    m.ParseFromString(f.read())

tokenizer.save("/Users/xigsun/Documents/repo/candle/candle-examples/examples/marian-mt/opus-mt-zh-en/tokenizer-marian-base-big.json")