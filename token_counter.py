
# To count number of tokens in retrieved passages
import nltk
from nltk.tokenize import word_tokenize
import json
with open("multilingual_output/xor_dev_retrieve_eng_span_v1_1_results.json") as f:
    content = f.read()

decoder = json.JSONDecoder()
predictions, _ = decoder.raw_decode(content.strip())

print(f"Total predictions loaded: {len(predictions)}")

for item in predictions[:3]:
    total_tokens = sum(len(word_tokenize(ctx)) for ctx in item["ctxs"])
    print(f"ID: {item['id']} | Passages: {len(item['ctxs'])} | Total tokens: {total_tokens}")