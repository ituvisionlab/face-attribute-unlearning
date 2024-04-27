import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from model import train
from metric import hamming_score

class Noise(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad = True)
        
    def forward(self):
        return self.noise
    
class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


class AdvancedNegGrad():
    def __init__(self, args, model, forget_dataloader, retain_dataloader, device, num_epochs=2):
        self.args = args
        self.model = model
        self.forget_dataloader_train = forget_dataloader
        self.retain_dataloader_train = retain_dataloader
        self.num_epochs = num_epochs
        self.device = device

    def unlearn(self):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)

        print_every = 20

        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0
            # Training on forget data with Gradient Ascent
            for batch_idx, (forget_batch, retain_batch) in enumerate(zip(self.forget_dataloader_train, self.retain_dataloader_train)):                
            
                forget_inputs, forget_labels = forget_batch.values()
                forget_inputs, forget_labels = forget_inputs.to(self.device), forget_labels.to(self.device)

                retain_inputs, retain_labels = retain_batch.values()
                retain_inputs, retain_labels = retain_inputs.to(self.device), retain_labels.to(self.device)

                outputs_forget = self.model(forget_inputs)
                outputs_retain = self.model(retain_inputs)


                # Gradient Ascent loss for forget data
                loss_ascent_forget = -criterion(outputs_forget, forget_labels)
                loss_retain = criterion(outputs_retain, retain_labels)
                overall_loss = loss_ascent_forget + loss_retain

                optimizer.zero_grad()
                overall_loss.backward()
                optimizer.step()
                running_loss += overall_loss.item() * forget_inputs.size(0)

                if (batch_idx + 1) % print_every == 0:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], Batch [{batch_idx+1}/{len(self.forget_dataloader_train)}] - Batch Loss: {loss_ascent_forget.item():.4f}")

            average_epoch_loss = running_loss / (len(self.forget_dataloader_train) * forget_inputs.size(0))
            print(f"Epoch [{epoch+1}/{self.num_epochs}] - Total Loss: {average_epoch_loss:.4f}")            

        return self.model

class NormalNegGrad():
    def __init__(self, args, model, forget_dataloader, retain_dataloader, device, num_epochs=2):
        self.args = args
        self.model = model
        self.forget_dataloader_train = forget_dataloader
        self.retain_dataloader_train = retain_dataloader
        self.num_epochs = num_epochs
        self.device = device

    def unlearn(self):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)

        print_every = 20

        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0
            # Training on forget data with Gradient Ascent
            for batch_idx, batch in enumerate(self.forget_dataloader_train):                
                inputs, labels = batch.values()
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs_forget = self.model(inputs)

                # Gradient Ascent loss for forget data
                loss_ascent_forget = -criterion(outputs_forget, labels)

                optimizer.zero_grad()
                loss_ascent_forget.backward()
                optimizer.step()
                running_loss -= loss_ascent_forget.item()

                if (batch_idx + 1) % print_every == 0:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], Batch [{batch_idx+1}/{len(self.forget_dataloader_train)}] - Batch Loss: {loss_ascent_forget.item():.4f}")

            average_epoch_loss = running_loss / (len(self.forget_dataloader_train))
            print(f"Epoch [{epoch+1}/{self.num_epochs}] - Total Loss: {average_epoch_loss:.4f}")

        return self.model

class Scrub():
    def __init__(self, student, teacher, forget_dataloader_train, retain_dataloader_train, device, num_epochs=10, max_steps=5):
        self.student = student # scrub_model -> the same model
        self.teacher = teacher # original_model -> to_be_unlearned_model
        self.retain_dataloader = retain_dataloader_train
        self.forget_dataloader = forget_dataloader_train
        self.num_epochs = num_epochs
        self.device = device
        self.max_steps = max_steps

    def unlearn(self):
        self.student.train()
        self.teacher.eval()

        for epoch in range(self.num_epochs):
            running_loss = []
            hamming_scores = []
            batch_losses = []

            optimizer = optim.Adam(self.student.parameters(), lr=0.0005, weight_decay=0.0005)
            criterion_cls = nn.BCEWithLogitsLoss()
            criterion_div = DistillKL(4.0)
            
            if epoch < self.max_steps:
                for i, batch in enumerate(self.forget_dataloader):
                    inputs, labels = batch.values()
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # Forward pass: Student (remove torch.no_grad() block for student)
                    optimizer.zero_grad()
                    outputs_student = self.student(inputs)
                    with torch.no_grad():
                        outputs_teacher = self.teacher(inputs)

                    loss = -criterion_div(outputs_student, outputs_teacher)*0.1

                    loss.backward()
                    optimizer.step()

                    running_loss.append(loss.item())

                    threshold = 0.5
                    predicted_labels = (outputs_student > threshold).float()

                    hamming_scores.append(hamming_score(labels.cpu().numpy(), predicted_labels.cpu().numpy()))
            else:
                for i, batch in enumerate(self.retain_dataloader):
                    inputs, labels = batch.values()
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs_student = self.student(inputs)
                    with torch.no_grad():
                        outputs_teacher = self.teacher(inputs)

                    loss_cls = criterion_cls(outputs_student, labels)
                    batch_losses.append(loss_cls.item())

                    loss_div_retain = criterion_div(outputs_student, outputs_teacher)
                    loss = loss_cls + loss_div_retain

                    loss.backward()
                    optimizer.step()

                    running_loss.append(loss.item())

                    threshold = 0.5
                    predicted_labels = (outputs_student > threshold).float()

                    hamming_scores.append(hamming_score(labels.cpu().numpy(), predicted_labels.cpu().numpy()))

            # Calculate average loss and accuracy
            avg_loss = torch.mean(torch.tensor(running_loss)).item()
            avg_accuracy = torch.mean(torch.tensor(hamming_scores)).item()
            print(f'[Loss: {avg_loss:.4f}, Average Training Accuracy: {avg_accuracy:.2f}%')
        
            print(f"Epoch {epoch+1} completed.")

        return self.student

class UNSIR():
    def __init__(self, args, model, classes_to_forget, metric_func, num_classes, retain_samples, log_folder, device, num_epochs=15, num_steps=8):
        self.args = args
        self.model = model
        self.classes_to_forget = classes_to_forget
        self.metric_func = metric_func
        self.num_classes = num_classes
        self.retain_samples = retain_samples
        self.log_folder = log_folder
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.device = device
        self.noises = {}

    def unlearn(self):
        self.learn_noise()
        return self.impair_and_repair_model()

    def learn_noise(self):
        noises = {}
        for cls in self.classes_to_forget:
            print("Optimizing loss for class {}".format(cls))

            noises[cls] = Noise(self.args.batch_size, 3, 64, 64).cuda()
            opt = torch.optim.Adam(noises[cls].parameters(), lr = 0.1)

            criterion = nn.BCEWithLogitsLoss()      
            for epoch in range(self.num_epochs):
                self.model.eval() # bec no training in this phase for the model itself
                
                running_loss = 0.0
                total_train_loss = 0

                total_loss = []
                hamming_score_list = []
                for batch in range(self.num_steps):
                    inputs = noises[cls]()
                    labels = torch.zeros(self.args.batch_size, self.num_classes).cuda()
                    labels[:, cls] = 1

                    opt.zero_grad()
                    outputs = self.model(inputs)

                    # loss = -F.cross_entropy(outputs, labels.long()) + 0.1*torch.mean(torch.sum(torch.square(inputs), [1, 2, 3]))

                    loss = -criterion(outputs, labels) + 0.1*torch.mean(torch.sum(torch.square(inputs), [1, 2, 3]))

                    loss.backward()
                    opt.step()

                    total_loss.append(loss.cpu().detach().numpy())

                    predicted_labels = (outputs > 0.5).float()
                    hamming_score_value = self.metric_func(labels.cpu().numpy(), predicted_labels.cpu().numpy())
                    hamming_score_list.append(hamming_score_value)

                print("Loss: {}".format(np.mean(total_loss)))

        self.noises = noises

    def impair_and_repair_model(self):
        noisy_data = []
        num_batches = 20

        for cls in self.classes_to_forget:
            for i in range(num_batches):
                batch = self.noises[cls]().cpu().detach()
                for i in range(batch.size(0)):
                    label = torch.zeros(40)
                    label[cls] = 1
                    noisy_data.append({'image':batch[i], 'label':label})

        noisy_data += self.retain_samples

        criterion = nn.BCEWithLogitsLoss() 
        
        metrics = {"hamming":self.metric_func}

        noisy_loader = torch.utils.data.DataLoader(noisy_data, batch_size=256, shuffle = True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.02)
        train_losses, train_scores, val_losses, val_scores = train(self.model, noisy_loader, noisy_loader, criterion, optimizer, 1, metrics, 1, self.device, self.args, self.log_folder, "unlearning_impair")

        heal_loader = torch.utils.data.DataLoader(self.retain_samples, batch_size=256, shuffle = True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.01)
        train_losses, train_scores, val_losses, val_scores = train(self.model, heal_loader, heal_loader, criterion, optimizer, 1, metrics, 1, self.device, self.args, self.log_folder, "unlearning_repair")

        return self.model

class BadT():
    def __init__(self):
        pass

    def unlearn(self, unlearning_teacher, student_model, model, retain_dataloader, forget_dataloader, device):
        optimizer = torch.optim.Adam(student_model.parameters(), lr = 0.0001)
        KL_temperature = 1

        self.blindspot_unlearner(model=student_model,
                                 unlearning_teacher = unlearning_teacher,
                                 full_trained_teacher = model, 
                                retain_dataloader = retain_dataloader,
                                forget_dataloader = forget_dataloader,
                                epochs=1, optimizer=optimizer, lr=0.0001, batch_size=256, num_workers=32,
                                device=device, KL_temperature=KL_temperature)
        
        return student_model

    def UnlearnerLoss(self, output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
        labels = torch.unsqueeze(labels, dim = 1)
        
        f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
        u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

        overall_teacher_out = labels * u_teacher_out + (1-labels)*f_teacher_out
        student_out = F.log_softmax(output / KL_temperature, dim=1)
        return F.kl_div(student_out, overall_teacher_out)
    
    def unlearning_step(self, model, unlearning_teacher, full_trained_teacher, forget_dataloader, retain_dataloader, optimizer, 
                device, KL_temperature):
        losses = []
        for batch in forget_dataloader:
            x, y = batch.values()
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                full_teacher_logits = full_trained_teacher(x)
                unlearn_teacher_logits = unlearning_teacher(x)
            output = model(x)
            optimizer.zero_grad()
            loss = self.UnlearnerLoss(output=output, labels=y, full_teacher_logits=full_teacher_logits, 
                                      unlearn_teacher_logits=unlearn_teacher_logits, KL_temperature=KL_temperature)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())
        
        for batch in retain_dataloader:
            x, y = batch.values()
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                full_teacher_logits = full_trained_teacher(x)
                unlearn_teacher_logits = unlearning_teacher(x)
            output = model(x)
            optimizer.zero_grad()
            loss = self.UnlearnerLoss(output=output, labels=y, full_teacher_logits=full_teacher_logits, 
                                      unlearn_teacher_logits=unlearn_teacher_logits, KL_temperature=KL_temperature)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())

        return np.mean(losses)


    def blindspot_unlearner(self, model, unlearning_teacher, full_trained_teacher, retain_dataloader, forget_dataloader, epochs = 10,
                optimizer = 'adam', lr = 0.01, batch_size = 256, num_workers = 32, 
                device = 'cuda', KL_temperature = 1):
        unlearning_teacher.eval()
        full_trained_teacher.eval()
        optimizer = optimizer
        if optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        else:
            # if optimizer is not a valid string, then assuming it as a function to return optimizer
            optimizer = optimizer#(model.parameters())

        for epoch in range(epochs):
            loss = self.unlearning_step(model = model, unlearning_teacher= unlearning_teacher, 
                            full_trained_teacher=full_trained_teacher, forget_dataloader=forget_dataloader, retain_dataloader=retain_dataloader, 
                            optimizer=optimizer, device=device, KL_temperature=KL_temperature)
            print("Epoch {} Unlearning Loss {}".format(epoch+1, loss))