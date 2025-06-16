import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")


def tokenize_dp(text):
    return tokenizer.encode_plus(
        text, max_length=375, padding="max_length", truncation=True, return_tensors="pt"
    )["input_ids"].squeeze(0)


class MultiOutputModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_config):
        super().__init__()
        self.embedding = nn.Embedding(tokenizer.vocab_size, embedding_dim)
        self.rnn = nn.LSTM(
            embedding_dim, hidden_dim, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(2 * hidden_dim, 512)

        self.heads = nn.ModuleDict()
        for idx, (name, type_, num_classes) in enumerate(output_config):
            if type_ == "categorical":
                self.heads[f"head_{idx}"] = nn.Linear(512, num_classes)
            elif type_ == "numerical":
                self.heads[f"head_{idx}"] = nn.Sequential(
                    nn.Linear(512, 1), nn.Sigmoid()
                )

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = torch.mean(x, dim=1)
        x = torch.relu(self.fc(x))

        outputs = []
        for head in self.heads.values():
            outputs.append(head(x))
        return torch.cat(outputs, dim=1)


class SymptomExtractionModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name="DeepPavlov/rubert-base-cased",
        numerical_features=None,
        categorical_features=None,
        categorical_num_classes=None,
    ):
        super(SymptomExtractionModel, self).__init__()

        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        hidden_size = self.bert.config.hidden_size

        self.numerical_features = numerical_features or []
        self.numerical_heads = nn.ModuleDict()
        self.numerical_missing_heads = nn.ModuleDict()
        for feature in self.numerical_features:
            self.numerical_heads[feature] = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )
            self.numerical_missing_heads[feature] = nn.Linear(
                hidden_size, 2
            )  # 0 = значение есть, 1 = пропуск

        self.categorical_features = categorical_features or []
        self.categorical_heads = nn.ModuleDict()
        for feature in self.categorical_features:
            num_classes = categorical_num_classes.get(feature, 2)
            self.categorical_heads[feature] = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs.last_hidden_state[:, 0, :]
        sequence_output = self.dropout(sequence_output)

        numerical_outputs = {
            feature: head(sequence_output)
            for feature, head in self.numerical_heads.items()
        }

        numerical_missing_outputs = {
            feature: head(sequence_output)
            for feature, head in self.numerical_missing_heads.items()
        }

        categorical_outputs = {
            feature: head(sequence_output)
            for feature, head in self.categorical_heads.items()
        }

        return numerical_outputs, numerical_missing_outputs, categorical_outputs
