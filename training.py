
df = load_and_preprocess_data()

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


# Separating the data set into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2)

# Creating custom Dataset classes
train_dataset = PoemDataset(train_df, tokenizer)
val_dataset = PoemDataset(val_df, tokenizer)

# Creating DataLoaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)
   
import numpy as np

class EarlyStopping:
    
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
        """Saves the model when the validation loss decreases."""
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

# Objective function for hyperparameter optimization
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
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward 
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )

        # Grandyant Algorithm
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Calculate training loss
    train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Training loss: {train_loss}")

    # Calculate verification loss
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

    # Check early stop condition
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("early stop condition")
        break

# After training, upload the best model
model.load_state_dict(torch.load('checkpoint.pt'))
        
# Model Validation and Tuning
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

# Save the model for later use
model.save_pretrained("./yol/talimat_sair_modelim")
tokenizer.save_pretrained("./yol/talimat_sair_tokenizerim")

from transformers import BartTokenizer, BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained("./yol/talimat_sair_modelim")
tokenizer = BartTokenizer.from_pretrained("./yol/talimat_sair_tokenizerim")

model.to(device)
model.eval()
