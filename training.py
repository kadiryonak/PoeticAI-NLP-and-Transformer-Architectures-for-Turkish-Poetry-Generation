
# Veriyi modele hazırlama adımı
class PoemDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.instructions = dataframe['instruction'].tolist()
        self.poems = dataframe['poem'].tolist()
        self.max_length = max_length

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        instruction = self.instructions[idx]
        poem = self.poems[idx]
        
        # Encoder girdisi olarak 'instruction' kullanılır
        encoder_inputs = self.tokenizer(
            instruction, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True,
            return_tensors="pt"
        )
        
        # Decoder girdisi olarak 'poem' kullanılır ve başına <bos> token'i eklenir
        decoder_inputs = self.tokenizer(
            poem, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True,
            add_special_tokens=True,  # BOS ve EOS token'lerini ekler
            return_tensors="pt"
        )
        
        # Şiir metnini labels olarak kullanırız, burada aynı zamanda <eos> token'i dahil edilir
        labels = decoder_inputs['input_ids'].squeeze()
        
        # Decoder input için, labels'dan farklı olarak, ilk token olarak <bos> token'ini kullanırız
        # Bunu yapabilmek için, labels'dan başlayarak bir token kaydırarak ve ilk token olarak <bos> ekleyerek yapabiliriz
        decoder_input_ids = torch.cat([
            torch.tensor([self.tokenizer.bos_token_id]),  # <bos> token'i eklenir
            labels[:-1]  # <eos> token'i hariç tutulur
        ])
        
        return {
            'input_ids': encoder_inputs['input_ids'].squeeze(),
            'attention_mask': encoder_inputs['attention_mask'].squeeze(),
            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
        }


from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Veri setini eğitim ve doğrulama setlerine ayırma
train_df, val_df = train_test_split(df, test_size=0.2)

# Özel Dataset sınıflarını oluşturma
train_dataset = PoemDataset(train_df, tokenizer)
val_dataset = PoemDataset(val_df, tokenizer)

# DataLoader'ları oluşturma
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)
   
import numpy as np

class EarlyStopping:
    """Eğitim sırasında erken durdurmayı uygulamak için bir sınıf."""
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt'):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Doğrulama kaybı azaldığında modeli kaydeder."""
        if self.verbose:
            print(f'Doğrulama kaybı azaldı ({self.val_loss_min:.6f} --> {val_loss:.6f}). Model kaydediliyor...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    
from transformers import AdamW
from torch.utils.data import DataLoader
import torch
import optuna

epochs = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparametre optimizasyonu için objective fonksiyonu
def objective(trial):
    
   
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    epochs = trial.suggest_int('epochs', 1, 5)

    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # Veri setleri ve DataLoader'lar
    train_dataset = PoemDataset(train_df, tokenizer)
    val_dataset = PoemDataset(val_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Training loss: {total_loss / len(train_loader)}")

    # Doğrulama kaybını hesapla
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
    return val_loss / len(val_loader)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("En iyi hiperparametreler: ", study.best_trial.params)
best_params = study.best_trial.params
# En iyi hiperparametreleri kullanarak modelinizi eğitin
lr = best_params['lr']
batch_size = best_params['batch_size']
epochs = best_params['epochs'] 

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
optimizer = AdamW(model.parameters(), lr= lr)
early_stopping = EarlyStopping(patience=3, verbose=True, path='checkpoint.pt')
model.to(device)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        # Cihaza gönder
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Modelin forward metodunu çağır
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )

        # Kayıp fonksiyonunu hesapla ve geri yayılımı uygula
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Eğitim kaybını hesapla
    train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Training loss: {train_loss}")

    # Doğrulama kaybını hesapla
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels
            )
            loss = outputs.loss
            val_loss += loss.item()
    
    val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}, Validation loss: {val_loss}")

    # Erken durdurma koşulunu kontrol et
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Erken durdurma gerçekleşti. Eğitim durduruluyor.")
        break

# Eğitim sonrası, en iyi modeli yükle
model.load_state_dict(torch.load('checkpoint.pt'))
        
# Model Doğrulama ve Ayarlama
model.eval()
total_eval_loss = 0

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )

        loss = outputs.loss
        total_eval_loss += loss.item()

print(f"Validation Loss: {total_eval_loss / len(val_loader)}")

# Modelin daha sonra kullanabilmek için kaydı
model.save_pretrained("./yol/talimat_sair_modelim")
tokenizer.save_pretrained("./yol/talimat_sair_tokenizerim")

from transformers import BartTokenizer, BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained("./yol/talimat_sair_modelim")
tokenizer = BartTokenizer.from_pretrained("./yol/talimat_sair_tokenizerim")

model.to(device)
model.eval()
