import torch
from torch import nn

class LinearModel(nn.Module):
    def __init__(self,input_size,num_classes):
        super(LinearModel,self).__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size,num_classes)
    
    def forward(self,feature):
        output = self.linear(feature.view(-1, self.input_size))
        return output

class CNN5(torch.nn.Module):

    def __init__(self, num_classes = 10, normalization = False):
        super(CNN5, self).__init__()
        self.normalization = normalization
        if self.normalization:
            self.gn0 = nn.GroupNorm(num_groups=16, num_channels=32)
            self.gn1 = nn.GroupNorm(num_groups=16, num_channels=64)
            self.gn2 = nn.GroupNorm(num_groups=16, num_channels=128)
            self.gn3 = nn.GroupNorm(num_groups=16, num_channels=256)
        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh(),
            )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 100, kernel_size=3, stride=1, padding=1),
            torch.nn.AvgPool2d(kernel_size=4, stride=4),
        )
        # torch.nn.init.xavier_uniform_(self.layer5.weight)

    def forward(self, x):
        x = self.layer0(x)
        if self.normalization:
            x = self.gn0(x)
        x = self.layer1(x)
        if self.normalization:
            x = self.gn1(x)
        x = self.layer2(x)
        if self.normalization:
            x = self.gn2(x)
        x = self.layer3(x)
        if self.normalization:
            x = self.gn3(x)
        y = self.layer4(x).view(x.size(0), -1)
        return y
    
import os
# os.environ['HF_DATASETS_OFFLINE '] = "1"
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, RobertaForSequenceClassification
from torch.nn import CrossEntropyLoss
import torch
from pathlib import Path

def create_roberta(label_to_id, num_lables):
    model = RobertaForSequenceClassification.from_pretrained('FacebookAI/roberta-base', num_labels = num_lables)
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in label_to_id.items()}
    return model

def create_gpt(data_dir=None, model = 'gpt2'):
    if data_dir is not None:
        if not os.path.isdir(Path(data_dir+'/tokenizer')):
            tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
            os.makedirs(Path(data_dir+'/tokenizer'))
            tokenizer.save_pretrained(Path(data_dir+'/tokenizer'))
        else:
            tokenizer = AutoTokenizer.from_pretrained(Path(data_dir+'/tokenizer'))
    else:
        tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    context_length = 128
    if data_dir is not None:
        data_dir = data_dir + '/'+ model
        if not os.path.isdir(Path(data_dir+'/model_cfg')):
            config = AutoConfig.from_pretrained(
                model,
                vocab_size=len(tokenizer),
                n_ctx=context_length,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                )
            os.makedirs(Path(data_dir+'/model_cfg'))
            config.save_pretrained(Path(data_dir+'/model_cfg'))
        else:
            config = AutoConfig.from_pretrained(Path(data_dir+'/model_cfg'))
    else:
        config = AutoConfig.from_pretrained(
            model,
            vocab_size=len(tokenizer),
            n_ctx=context_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            )
    model = GPT2LMHeadModel(config)
    keytoken_ids = []
    for keyword in [
        "plt",
        "pd",
        "sk",
        "fit",
        "predict",
        " plt",
        " pd",
        " sk",
        " fit",
        " predict",
    ]:
        ids = tokenizer([keyword]).input_ids[0]
        if len(ids) == 1:
            keytoken_ids.append(ids[0])
        else:
            print(f"Keyword has not single token: {keyword}")
    return model, keytoken_ids

def keytoken_weighted_loss(inputs, logits, keytoken_ids, alpha=1.0):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Resize and average loss per sample
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    # Calculate and scale weighting
    weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(
        axis=[0, 2]
    )
    weights = alpha * (1.0 + weights)
    # Calculate weighted average
    weighted_loss = (loss_per_sample * weights).mean()
    return weighted_loss
