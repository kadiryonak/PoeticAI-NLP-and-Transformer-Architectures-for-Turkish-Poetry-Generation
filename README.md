# PoeticAI-NLP-and-Transformer-Architectures-for-Turkish-Poetry-Generation

Overview
PoeticAI harnesses the power of Natural Language Processing (NLP) and state-of-the-art Transformer architectures to create Turkish poetry. This project is rooted in Python and utilizes a suite of libraries such as Transformers, Datasets, Pandas, NumPy, and PyTorch. The core of PoeticAI is the implementation of the BART For Conditional Generation model, a transformer-based model fine-tuned to generate poetry. By incorporating advanced NLP techniques and leveraging transformer models, PoeticAI offers a novel approach to understanding and generating Turkish poetry automatically.

Project Structure
The project is meticulously organized into distinct sections, each dedicated to a specific phase of the poetry generation process:

Data Preprocessing (data_processing.py): Prepares the dataset for training by performing tasks such as loading, cleaning, and encoding the text data, ensuring it's in the right format for the model.

Model Configuration and Setup (model.py): Configures the transformer-based BART model for poetry generation. This script sets up the model architecture and readies it for the training phase.

Training and Hyperparameter Optimization (training.py): This script fine-tunes the BART model on the Turkish poetry dataset. It includes mechanisms for early stopping to avert overfitting and employs Optuna for the optimization of hyperparameters, aiming to achieve peak model performance.

Inference and Poem Generation (inference.py): Demonstrates the application of the trained model to generate new poems. This script showcases the model's capability to produce creative and contextually coherent Turkish poetry based on given prompts.

Main Execution Script (main.py): Serves as the project's entry point, orchestrating the entire process from data preprocessing and model training to the generation of new poetry.

Dependencies
The following Python libraries are essential for running PoeticAI. Ensure they are installed in your environment:

pandas
numpy
transformers
datasets
torch
optuna
Installation
Begin by cloning the PoeticAI repository to your local machine:

bash
Copy code
git clone <repository-url>
Then, navigate to the project directory and install the necessary dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Execute the main.py script within the src directory to start the project:

bash
Copy code
cd src
python main.py
The script will guide you through generating poetry or conducting other tasks defined within the project.

Contributing
We welcome contributions to PoeticAI! For guidelines on contributing, please consult the CONTRIBUTING.md file.

License
PoeticAI is released under the MIT license. For more details, refer to the LICENSE file.
