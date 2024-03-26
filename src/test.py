import torch
from pathlib import Path
from train import get_model
from utils.data import create_testds
import pickle
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def test_acc(model, test_dl):
    model.eval()
    correct_predictions = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batchidx, batch in enumerate(test_dl):
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            label = batch['label'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            classification_output = model.project(encoder_output)
            model_output = torch.argmax(classification_output, dim=-1)

            if(batchidx%80==0):
                print()
                print(model_output, label)

            correct_predictions += torch.sum(model_output.int() == label.int()).item()
            y_true.extend(label.cpu().numpy())
            y_pred.extend(model_output.cpu().numpy())

    accuracy = correct_predictions / len(test_dl)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    confusion = confusion_matrix(y_true, y_pred)
    
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print('Confusion Matrix:')
    print(confusion)

with open('config.pkl', 'rb') as f:
    config = pickle.load(f)

test_dataloader, tokenizer = create_testds(config)
model = get_model(config, tokenizer.get_vocab_size())

model_filename = config['best_weight_path']
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])

test_acc(model, test_dataloader)
