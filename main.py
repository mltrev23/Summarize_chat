from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

hugginface_dataset_name = 'knkarthick/dialogsum'

dataset = load_dataset(hugginface_dataset_name)

example_indices = [40, 200]

dash_line = '-'.join('' for x in range(100))

for i, index in enumerate(example_indices):
    print(dash_line)
    print(f'Example {i + 1}')
    print(dash_line)
    print('Input dialog: ')
    print(dataset['test'][index]['dialogue'])
    print(dash_line)
    print('Baseline human summary: ')
    print(dataset['test'][index]['summary'])
    print(dash_line)
    print()

model_name = 'google/flan-t5-base'

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)

sentence = 'What time is it, Tom?'

sentence_encoded = tokenizer(sentence, return_tensors = 'pt')

sentence_decoded = tokenizer.decode(
    sentence_encoded['input_ids'][0],
    skip_special_tokens = True
)

print('Encoded sentence:')
print(sentence_encoded['input_ids'][0])
print('\nDecoded sentence:')
print(sentence_decoded)