import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report
from sklearn import linear_model, model_selection

def get_outputs(model, val_dataloader, device):
    outputs = []

    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            inputs, labels = batch['image'], batch['label']
            inputs, labels = inputs.to(device), labels.to(device)

            batch_outputs = model(inputs)
            outputs.append(batch_outputs.cpu().detach())

    return outputs


def jsd_eval(p_model_outputs, q_model_outputs):
    jsd_list = []
    for p_model_batch_output, q_model_batch_output in zip(p_model_outputs, q_model_outputs):
        p_outputs = F.log_softmax(p_model_batch_output)
        q_outputs = F.log_softmax(q_model_batch_output)

        kl_p_q = F.kl_div(input=p_outputs, target=q_outputs, log_target=True, reduction='mean')
        kl_q_p = F.kl_div(input=q_outputs, target=p_outputs, log_target=True, reduction='mean')
        jsd = 0.5*kl_p_q + 0.5*kl_q_p

        jsd_list.append(jsd)

    jsd_res = torch.stack(jsd_list).mean()   
    return jsd_res.item()


def calculate_jsd(retrained_model, unlearned_model, dataloader, device):
    retrained_outputs = get_outputs(retrained_model, dataloader, device)
    unlearned_outputs = get_outputs(unlearned_model, dataloader, device)

    return jsd_eval(retrained_outputs, unlearned_outputs)


def hamming_score(y_true, y_pred):
    num_samples = len(y_true)
    total_correct = 0
    
    for true_labels, pred_labels in zip(y_true, y_pred):
        correct_labels = (true_labels == pred_labels).sum()
        total_correct += correct_labels
    hamming_score = total_correct / (num_samples * len(y_true[0]))
    return hamming_score


def multi_label_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


def compute_losses(model, loader, device, size=None):
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    all_losses = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch.values()
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            losses = criterion(outputs, labels).mean(dim=1).cpu().detach().numpy()
            for l in losses:
                all_losses.append(l)
            
            if size != None and len(all_losses) >= size:
                break


    return np.array(all_losses)