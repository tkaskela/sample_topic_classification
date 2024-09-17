from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("NEU-HAI/Llama-2-7b-alpaca-cleaned")
model = AutoModelForCausalLM.from_pretrained("NEU-HAI/Llama-2-7b-alpaca-cleaned", force_download=True)

text = input("Enter a sentence: ")