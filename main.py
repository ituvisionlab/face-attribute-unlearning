from arg import parse_args
from model import *
from dataset import *
from metric import *
from plot import *
from unlearning import *

import torch.nn as nn
import torch
import uuid
from datetime import datetime
import os
import json
import random
import copy

import warnings
warnings.filterwarnings("ignore")


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(out_args=None, seed=1773):
    if out_args == None:
        args, _ = parse_args()
    else:
        args = Namespace(**out_args)

    print(vars(args))

    _temp = [3, 6, 14]
    classes_to_forget = _temp[:args.n_class_forget]

    set_seed(seed)

    unique_id = str(uuid.uuid4())[:8]

    log_folder = "./logs"
    if args.do_train:
        log_folder += f"/trained_on_{args.train_data}_wo_unlearning"
        if args.train_data == 'remain-train':
            log_folder += f"/{args.n_class_forget}_class_forget"
    if args.do_unlearn:
        log_folder += f"/{args.method}/{args.n_class_forget}_class_forget"
    
    if args.do_finetune:
        log_folder += f"/finetuned_on_remain-train_wo_unlearning/{args.n_class_forget}_class_forget"

    log_folder += f"/{args.model_arch}_seed_{seed}_{datetime.now().strftime("%d.%m-%H:%M")}"
    os.makedirs(log_folder)

    device = torch.device(args.device)

    train_dataset, val_dataset, test_dataset = get_dataset(args.dataset, args.dataset_path)
    
    train_loader, val_loader, test_loader = get_loader(train_dataset, val_dataset, test_dataset, batch_size=args.batch_size, device=device)
    retain_dl, forget_dl, retain_val_dl, forget_val_dl, retain_test_dl, forget_test_dl, retain_samples = get_forget_and_retain_dataloaders(train_dataset, val_dataset, test_dataset, args, classes_to_forget)

    n_classes = train_dataset.n_classes

    #
    model = get_model(args.model_arch, args.is_pretrained, args.n_features, args.dropout_prob, n_classes, device)
    
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    metrics = {
        "hamming": hamming_score
    }

    train_losses, train_scores, val_losses, val_scores = [], [], [], []
    if args.do_train:
        if args.train_data == 'all-train':
            tl, vl = train_loader, val_loader
        elif args.train_data == 'forget-train':
            tl, vl = forget_dl, forget_val_dl
        elif args.train_data == 'remain-train':
            tl, vl = retain_dl, retain_val_dl

        train_losses, train_scores, val_losses, val_scores = train(model, tl, vl, criterion, optimizer, args.max_epoch, metrics, args.eval_per_epoch, device, args, log_folder, "model_training")

    if args.do_finetune:
        if args.do_train == False:
            model = model_load(args.model_arch, args.is_pretrained, args.n_features, args.dropout_prob, n_classes, device, args.model_path)
            tl, vl = retain_dl, retain_val_dl
            train_losses, train_scores, val_losses, val_scores = train(model, tl, vl, criterion, optimizer, args.max_epoch, metrics, args.eval_per_epoch, device, args, log_folder, "model_finetuning")
    
    if args.do_unlearn:
        if args.do_train == False:
            model = model_load(args.model_arch, args.is_pretrained, args.n_features, args.dropout_prob, n_classes, device, args.model_path)
            
            if args.method == "unsir":
                unlearning_method = UNSIR(args, model, classes_to_forget, metrics["hamming"], n_classes, retain_samples, log_folder, device)
                model = unlearning_method.unlearn()
            elif args.method == "normal-neggrad":
                unlearning_method = NormalNegGrad(args, model, forget_dl, retain_dl, device, num_epochs=2)
                model = unlearning_method.unlearn()
            elif args.method == "advanced-neggrad":
                unlearning_method = AdvancedNegGrad(args, model, forget_dl, retain_dl, device, num_epochs=2)
                model = unlearning_method.unlearn()
            elif args.method == "scrub":
                student = model
                teacher = copy.deepcopy(model)
                unlearning_method = Scrub(student, teacher, forget_dl, retain_dl, device, num_epochs=10, max_steps=5)
                model = unlearning_method.unlearn()
            elif args.method == "badt":
                unlearning_teacher = get_model(args.model_arch, False, args.n_features, args.dropout_prob, n_classes, device)
                student_model = copy.deepcopy(model)

                unlearning_method = BadT()
                model = unlearning_method.unlearn(unlearning_teacher, student_model, model, retain_dl, forget_dl, device)

    

    test_scores_all_test, test_scores_retain_test, test_scores_forget_test = [], [], []
    classify_report_all_test, classify_report_retain_test, classify_report_forget_test = None, None, None
    jsd_score_all_test, jsd_score_forget_test, jsd_score_retain_test = None, None, None
    if args.do_test:
        test_scores_all_test, classify_report_all_test = test(model, test_loader, criterion, metrics, device, args)
        test_scores_retain_test, classify_report_retain_test = test(model, retain_test_dl, criterion, metrics, device, args)
        test_scores_forget_test, classify_report_forget_test = test(model, forget_test_dl, criterion, metrics, device, args)

        if args.do_unlearn or args.do_finetune:
            retrained_model = model_load(args.model_arch, False, args.n_features, args.dropout_prob, 40, device, model_path=args.retrain_model_path)

            jsd_score_all_test = calculate_jsd(retrained_model, model, test_loader, device)
            jsd_score_retain_test = calculate_jsd(retrained_model, model, retain_test_dl, device)
            jsd_score_forget_test = calculate_jsd(retrained_model, model, forget_test_dl, device)



    if train_scores != []:
        train_scores = {key: [value.item() for value in values] for key, values in train_scores.items()}
    if val_scores != []:
        val_scores = {key: [value.item() for value in values] for key, values in val_scores.items()}
    if test_scores_all_test != []:
        test_scores_all_test = {key: value.item() for key, value in test_scores_all_test.items()}
        test_scores_retain_test = {key: value.item() for key, value in test_scores_retain_test.items()}
        test_scores_forget_test = {key: value.item() for key, value in test_scores_forget_test.items()}

    result_dict = {
        "Time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "Args": vars(args),
        "Train Loss": train_losses,
        "Val Loss": val_losses,
        "Train Scores": train_scores,
        "Val Scores": val_scores,
        "Classify Report All": classify_report_all_test,
        "Classify Report Retain": classify_report_retain_test,
        "Classify Report Forget": classify_report_forget_test,
        "JSD Score All": jsd_score_all_test,
        "JSD Score Retain": jsd_score_retain_test,
        "JSD Score Forget": jsd_score_forget_test,
        "Test Score All": test_scores_all_test,
        "Test Score Retain": test_scores_retain_test,
        "Test Score Forget": test_scores_forget_test
    }

    with open(f"{log_folder}/{unique_id}_exp.json", "w") as file:
        json.dump(result_dict, file)
    
    loss_fig_path = f"{log_folder}/loss_plot.png"
    save_loss_plot(train_losses, val_losses, loss_fig_path)

    try:
        torch.save(model.state_dict(), f"{log_folder}/final_model.pth")
    except:
        print("MODEL COULD NOT BE SAVED!")

    
if __name__ == "__main__":
    main()