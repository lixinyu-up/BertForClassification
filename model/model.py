import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)



class JointBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst):
        super(JointBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.bert = BertModel(config=config)

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)


    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        pooled_output = outputs[1]  # [CLS]
        intent_logits = self.intent_classifier(pooled_output)

        total_loss = 0
        # Intent Softmax
        if intent_label_ids is not None:
            # training
            loss = nn.CrossEntropyLoss()
            total_loss += loss(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))

        outputs = (total_loss, intent_logits) + outputs[2:]
        return outputs  # loss, logits, (hidden_states), (attentions) 