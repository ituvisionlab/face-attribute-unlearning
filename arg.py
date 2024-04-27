import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        default='celeba', type=str, required=False, help="celeba or another dataset")
    parser.add_argument("--dataset_path",   default='celeba', type=str, required=False, help="celeba or another dataset")

    parser.add_argument("--model_arch",     default="resnet18", type=str, required=False, help="'resnet18' or 'resnet50' or 'vit'")
    parser.add_argument("--is_pretrained",  action="store_true", help="init model with pretrained weights")
    
    parser.add_argument("--method",         default="badt", type=str, required=False, help="unlearning method unsir or normal-neggrad or advanced-neggrad or scrub or badt")

    parser.add_argument("--train_data",     default="remain-train", type=str, required=False, help="'all-train' or 'forget-train' or 'remain-train'")
    parser.add_argument("--finetune_data",  default="test", type=str, required=False, help="'all-train' or 'forget-train' or 'remain-train'")

    parser.add_argument("--do_train",       action="store_true", help="Whether to run training.")
    parser.add_argument("--do_test",        action="store_false", help="Whether to run eval on the test set.")
    parser.add_argument("--do_finetune",    action="store_true", help="Whether to finetune")
    parser.add_argument("--do_unlearn",     action="store_false", help="Whether to run unlearning.")
    
    parser.add_argument('--n_class_forget',     type=str, default=1, help='number of classes to forget')
    parser.add_argument("--model_path",         type=str, required=False, help="path to trained model file, which is trained on all-train")
    parser.add_argument("--retrain_model_path", type=str, required=False, help="path to retrained model file")
    parser.add_argument("--save_path",      type=str, required=False, help="path to save the model")
    
    parser.add_argument("--seed",           default=1773, type=int, required=False, help="seed for random number generator")
    
    parser.add_argument("--n_features",     default=1028, type=int, help="init model with pretrained weights")
    parser.add_argument("--dropout_prob",   default=0.2, type=float, help="init model with pretrained weights")
    parser.add_argument("--device",             default='cuda', type=str, required=False, help="device")
    parser.add_argument("--batch_size",         default=256, type=int, required=False, help="training batch size, default 8")
    parser.add_argument("--eval_per_epoch",     default=1, type=int, required=False, help="eval_per_epoch")
    parser.add_argument("--save_model_per_epoch", default=1000, type=int, required=False, help="save_model_per_epoch")
    parser.add_argument("--max_epoch",          default=30, type=int, required=False, help="maximum epoch numbers")
    parser.add_argument("--weight_decay",       default=0.01, type=float, required=False, help="AdamW weight decay, default 0.01")
    parser.add_argument("--learning_rate",      default=1e-4, type=float, required=False, help="AdamW learning rate, default 5e-5")
    parser.add_argument("--adam_epsilon",       default=1e-8, type=float, required=False, help="AdamW epsilon, default 1e-8")

    args, _ = parser.parse_known_args()
    return args, parser