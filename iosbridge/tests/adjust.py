from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianTokenizer,MarianMTModel

modelpath = "opus-mt-zh-en"
modelpath.split("-")[2]
path = "/Users/xigsun/Documents/repo/mt-language/" + modelpath

model = AutoModelForSeq2SeqLM.from_pretrained(path)
model.save_pretrained(path + "/newpath", save_format="safetensors")
from convert_slow_tokenizer import MarianConverter
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
fast_tokenizer = MarianConverter(tokenizer, index=0).converted()
fast_tokenizer.save(path + "/tokenizer-marian-base-" + modelpath.split("-")[2] + ".json")
fast_tokenizer = MarianConverter(tokenizer, index=1).converted()
fast_tokenizer.save(path + "/tokenizer-marian-base-" + modelpath.split("-")[3] + ".json")


