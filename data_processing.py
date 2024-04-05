
from datasets import load_dataset
import pandas as pd
from transformers import BartTokenizer

def load_and_prepare_data(tokenizer_model='facebook/bart-large-cnn', max_length=512):
    # Dataset
    dataset = load_dataset("beratcmn/instruction-turkish-poems")
    
    # Pull 'instruction' and 'poem' columns from dataset
    poems_dataset = dataset['train'].map(lambda example: {'instruction': example['instruction'], 'poem': example['poem']})
    
    df = pd.DataFrame(poems_dataset)
    df = df[['instruction', 'poem']]
    
    # Install Tokenizer
    tokenizer = BartTokenizer.from_pretrained(tokenizer_model)
    
    # Data preparation function
    def prepare_data(instruction, poem):
        # Encoder input (Komutları tokenleştirme)
        encoder_inputs = tokenizer.encode_plus(
            instruction.lower(),  # Metni küçük harfe çevirme
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Decoder input (Tokenizing poems)
        decoder_inputs = tokenizer.encode_plus(
            poem.lower(),  
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoder_inputs['input_ids'].squeeze(),
            'attention_mask': encoder_inputs['attention_mask'].squeeze(),
            'decoder_input_ids': decoder_inputs['input_ids'].squeeze(),
            'decoder_attention_mask': decoder_inputs['attention_mask'].squeeze(),
            'labels': decoder_inputs['input_ids'].squeeze()
        }

    # Data preparation on the entire data set
    prepared_data = [prepare_data(row['instruction'], row['poem']) for _, row in df.iterrows()]
    
    return prepared_data

# Load the tokenizer model and get the prepared data
tokenizer_model = 'facebook/bart-large-cnn'
prepared_data = load_and_prepare_data(tokenizer_model)

# Check the data for the first five examples
for data in prepared_data[:5]:
    print("Encoder Input IDs:", data['input_ids'])
    print("Decoder Input IDs:", data['decoder_input_ids'])
    print("---")
