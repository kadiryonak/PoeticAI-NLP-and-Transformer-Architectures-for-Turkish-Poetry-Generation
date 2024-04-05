from transformers import BartTokenizer, BartForConditionalGeneration
import torch

def generate_poems(instruction):
    # Model ve tokenizer yükleme
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer()
    model.to(device)
    model.eval()
    
   
    inputs = tokenizer(instruction, return_tensors="pt").to(device)
    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=100,
        temperature=0.9,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=3
    )
    
    
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        print(f"=== Generated Poem {generated_sequence_idx + 1} ===")
        text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        print(text, "\n")
