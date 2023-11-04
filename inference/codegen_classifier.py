from transformers import AutoTokenizer, AutoModelForCausalLM, CodeGenPreTrainedModel
import torch, torch.nn as nn
import copy
from typing import Optional, Tuple, Union

class CodegenClassifier(nn.Module):
    def __init__(self, codegen_model):
        super().__init__()
        self.transformer = copy.deepcopy(codegen_model.get_transformer()).to("cuda")
        self.embed_dim = codegen_model.get_lm_head().in_features
        self.classifier_head = nn.Linear(self.embed_dim, 1).to("cuda")
    
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
        ):
        transformer_outputs = self.transformer(
            input_ids,
            use_cache=use_cache,
        )
        hidden_states = transformer_outputs[0][:, -1, :]  # [1, 1024]
        classifier_logits = self.classifier_head(hidden_states).to(torch.float32)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCELoss()
            loss = loss_fct(classifier_logits, labels)
            loss = loss.to(hidden_states.dtype)
        return (loss, classifier_logits)
    
    
    def train(self, dataloader, optimizer, num_epochs=10, device='cuda'):
        self = self.to(device)
        self.train()  # set the model to train mode
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                optimizer.zero_grad()  # reset gradients
                loss, _ = self(input_ids=input_ids, labels=labels)  # forward pass
                loss.backward()  # compute gradients
                optimizer.step()  # update weights
                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)  # get average loss
            print(f'Epoch {epoch+1}/{num_epochs}: loss={epoch_loss}')