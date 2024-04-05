
from datasets import load_dataset
import pandas as pd
from transformers import BartTokenizer

def load_and_prepare_data(tokenizer_model='facebook/bart-large-cnn', max_length=512):
    # Veri setini yükleyin
    dataset = load_dataset("beratcmn/instruction-turkish-poems")
    
    # Veri setinden 'instruction' ve 'poem' sütunlarını çekin
    poems_dataset = dataset['train'].map(lambda example: {'instruction': example['instruction'], 'poem': example['poem']})
    
    df = pd.DataFrame(poems_dataset)
    df = df[['instruction', 'poem']]
    
    # Tokenizer'ı yükleyin
    tokenizer = BartTokenizer.from_pretrained(tokenizer_model)
    
    # Veri hazırlama fonksiyonu
    def prepare_data(instruction, poem):
        # Encoder input (Komutları tokenleştirme)
        encoder_inputs = tokenizer.encode_plus(
            instruction.lower(),  # Metni küçük harfe çevirme
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Decoder input (Şiirleri tokenleştirme)
        decoder_inputs = tokenizer.encode_plus(
            poem.lower(),  # Metni küçük harfe çevirme
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

    # Tüm veri seti üzerinde veri hazırlığı
    prepared_data = [prepare_data(row['instruction'], row['poem']) for _, row in df.iterrows()]
    
    return prepared_data

# Tokenizer modelini yükleyip hazırlanan veriyi al
tokenizer_model = 'facebook/bart-large-cnn'
prepared_data = load_and_prepare_data(tokenizer_model)

# İlk beş örneğin verilerini kontrol et
for data in prepared_data[:5]:
    print("Encoder Input IDs:", data['input_ids'])
    print("Decoder Input IDs:", data['decoder_input_ids'])
    print("---")