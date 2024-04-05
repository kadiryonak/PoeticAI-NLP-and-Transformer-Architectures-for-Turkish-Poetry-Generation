from data_processing import load_and_preprocess_data
from model import load_model_and_tokenizer
from training import train_model, PoemDataset
from inference import generate_poems

def main():
    df = load_and_preprocess_data()
    model, tokenizer = load_model_and_tokenizer()
    PoemDataset(prepared_data)
    """ 
     This section will be inherited from the training.py page
     and the training.py page will be arranged in accordance with main.py.  
    """
    train_model(df, tokenizer)
    instruction = "Güneşin batışı"
    generate_poems(instruction)

if __name__ == "__main__":
    main()
