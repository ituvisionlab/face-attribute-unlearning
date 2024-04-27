import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from tqdm import tqdm
import numpy as np
import timm

from metric import *


class ViTModel(torch.nn.Module):
    def __init__(self, num_features, dropout_prob, n_classes):
        super(ViTModel, self).__init__()

        self.model = timm.create_model('vit_small_patch16_224.augreg_in21k', pretrained=True)
        self.model.head = nn.Sequential(
            nn.Linear(self.model.head.in_features, num_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(num_features, n_classes)
        )
    
    def forward(self, inputs):
        inputs = F.interpolate(inputs, size=(224, 224), mode="bilinear")
        return self.model(inputs)



def get_model(model_arch, is_pretrained, num_features, dropout_prob, n_classes, device):
    if model_arch == "resnet50":
        if is_pretrained:
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            model = resnet50(weights=None)

        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, num_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(num_features, n_classes)
        )

    elif model_arch == "resnet18":
        if is_pretrained:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            model = resnet18(weights=None)

        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, num_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(num_features, n_classes)
        )

    elif model_arch == "vit":
        model = ViTModel(num_features, dropout_prob, n_classes)

    model = model.to(device)
    return model


def model_load(model_arch, is_pretrained, num_features, dropout_prob, n_classes, device, model_path):
    model = get_model(model_arch, is_pretrained, num_features, dropout_prob, n_classes, device).to('cpu')

    if model_arch == "vit":
        weights = torch.load(open(model_path, "rb"))
        model.load_state_dict(weights)
    else:
        model.load_state_dict(torch.load(model_path))
        
    return model.to(device)

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, score_metrics, eval_per_epoch, device, args, log_folder, train_name):
    '''
    score_metrics : {'hamming':func1, 'f1':func2, ...}
    
    '''

    train_losses = []
    val_losses = []

    train_scores = {}
    for metric_name in score_metrics.keys():
        train_scores[metric_name] = []
        
    val_scores = {}
    for metric_name in score_metrics.keys():
        val_scores[metric_name] = []


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        train_scores_epoch = {}
        for metric_name in score_metrics.keys():
            train_scores_epoch[metric_name] = []

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
        for i, batch in enumerate(train_bar):            
            inputs, labels = batch.values()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            predicted_labels = (outputs > 0.5).float()
            
            for metric_name, metric_func in score_metrics.items():
                train_scores_epoch[metric_name].append(metric_func(labels.cpu().numpy(), predicted_labels.cpu().numpy()))
            
            train_bar.set_postfix(loss=loss.item())
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        for metric_name in score_metrics.keys():
            train_scores[metric_name].append(torch.mean(torch.tensor(train_scores_epoch[metric_name])))


        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f} ", end="")
        for metric_name in score_metrics.keys():
            print(f"Train {metric_name} score: {train_scores[metric_name][-1]:.4f}", end=",")

        # evaluation
        if epoch % eval_per_epoch == 0:
            model.eval()
            total_val_loss = 0

            val_scores_epoch = {}
            for metric_name in score_metrics.keys():
                val_scores_epoch[metric_name] = []
            
            with torch.no_grad():
                for batch in val_loader:
                    inputs, labels = batch.values()
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item() * inputs.size(0)

                    predicted_labels = (outputs > 0.5).float()

                    for metric_name, metric_func in score_metrics.items():
                        val_scores_epoch[metric_name].append(metric_func(labels.cpu().numpy(), predicted_labels.cpu().numpy()))

            val_loss = total_val_loss / len(val_loader.dataset)
            val_losses.append(val_loss)

            for metric_name in score_metrics.keys():
                val_scores[metric_name].append(torch.mean(torch.tensor(val_scores_epoch[metric_name])))
            
            for metric_name in score_metrics.keys():
                print(f"Val {metric_name} score: {val_scores[metric_name][-1]:.4f}", end=",")
        
        name = f"{args.model_arch}_{args.train_data}"
        if epoch % args.save_model_per_epoch == 0 or epoch == num_epochs - 1:
            try:
                torch.save(model.state_dict(), f"{log_folder}/epoch_{epoch}.pth")
            except:
                print("MODEL COULDNT BE SAVED!!!!!!!!!!!!!!!!!!!!!")
        

    return [train_losses, train_scores, val_losses, val_scores]


def test(model, test_loader, criterion, score_metrics, device, args):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_true_labels = []
    all_predicted_labels = []

    test_scores = {}
    for metric_name in score_metrics.keys():
        test_scores[metric_name] = []

    test_bar = tqdm(test_loader, desc='Testing', unit='batch')
    with torch.no_grad():
        for batch in test_bar:
            inputs, labels = batch.values()
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            predicted_labels = (outputs > 0.5).float().cpu().numpy()
            true_labels = labels.cpu().numpy()

            for metric_name, metric_func in score_metrics.items():
                test_scores[metric_name].append(metric_func(labels.cpu().numpy(), predicted_labels))

            total_samples += inputs.size(0)
            all_true_labels.append(true_labels)
            all_predicted_labels.append(predicted_labels)

    true_labels = np.concatenate(all_true_labels, axis=0)
    predicted_labels = np.concatenate(all_predicted_labels, axis=0)

    average_loss = total_loss / total_samples

    avg_scores = {}
    for metric_name in score_metrics.keys():
        avg_scores[metric_name] = torch.mean(torch.tensor(test_scores[metric_name]))

    class_report_res = multi_label_classification_report(true_labels, predicted_labels)

    print(f"Test Loss: {average_loss:.4f}")

    for metric_name in score_metrics.keys():
        print(f"Val {metric_name} score: {avg_scores[metric_name]:.4f}", end=",")

    return avg_scores, class_report_res