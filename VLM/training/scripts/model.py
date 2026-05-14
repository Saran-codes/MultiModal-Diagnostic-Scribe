import torch
import torch.nn as nn
from transformers import BioGptForCausalLM
import torchvision.models as models

class VisionEncoder(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.model = models.resnet50(weights=None)
        # Remove pooling and classifier to get spatial feature maps
        self.model.avgpool = nn.Identity()
        self.model.fc = nn.Identity()

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            raw_weights = ckpt.get('state_dict', ckpt)
            
            # Clean prefixes from potential DistributedDataParallel or previous wrappers
            state_dict = {k.replace('model.', '').replace('backbone.', ''): v for k, v in raw_weights.items()}
            
            # Load weights safely
            msg = self.model.load_state_dict(state_dict, strict=False)
            print(f"--- Vision Load Status: {msg} ---")

    def forward(self, x):
        x = x.to(self.model.conv1.weight.dtype)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x) 
        # Output: [Batch, 2048, 32, 32]
        return x

class CytologyVLM(nn.Module):
    def __init__(self, vision_encoder, llm): 
        super().__init__()
        self.vision_encoder = vision_encoder
        self.llm = llm
        
        hidden_dim = llm.config.hidden_size 
        
        # Pool to 16x16 = 256 visual tokens
        self.pooler = nn.AdaptiveAvgPool2d((16, 16))
        # MLP bridge: projects ResNet (2048) to BioGPT hidden dim (1024)
        self.visual_projection = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def freeze_backbone(self):
        for param in self.llm.parameters():
            param.requires_grad = False
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        print("--- LLM and Vision Encoder Frozen. ---")

    def unfreeze_for_alignment(self, num_layers=4, unfreeze_vision=True):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.visual_projection.parameters():
            param.requires_grad = True

        total_layers = len(self.llm.biogpt.layers)
        for i in range(total_layers - num_layers, total_layers):
            for param in self.llm.biogpt.layers[i].parameters():
                param.requires_grad = True

        if unfreeze_vision:
            for param in self.vision_encoder.model.layer4.parameters():
                param.requires_grad = True
            print(f"Unfrozen: Bridge + last {num_layers} BioGPT layers + ResNet layer4")

    def forward(self, images, input_ids, attention_mask=None, output_attentions=False):
        features = self.vision_encoder(images)
        pooled_features = self.pooler(features)  # [B, 2048, 16, 16]
        visual_tokens = pooled_features.flatten(2).transpose(1, 2)  # [B, 256, 2048]
        visual_embeds = self.visual_projection(visual_tokens) * 1.2

        text_embeds = self.llm.biogpt.embed_tokens(input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            prefix_len = visual_embeds.shape[1]
            prefix_mask = torch.ones((batch_size, prefix_len), device=attention_mask.device)
            full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            full_mask = None

        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=full_mask, output_attentions=output_attentions)
        
        if output_attentions:
            return outputs.logits, outputs.attentions
        return outputs.logits
        
    def generate(self, images, prompt_ids, max_new_tokens, tokenizer, temperature=0.7, repetition_penalty=1.2):
        self.eval()
        with torch.no_grad():
            features = self.vision_encoder(images)
            pooled = self.pooler(features)
            visual_embeds = self.visual_projection(pooled.flatten(2).transpose(1, 2)) * 1.2

            prefix_embeds = visual_embeds
            curr_input_ids = prompt_ids
            finished = torch.zeros(images.size(0), dtype=torch.bool, device=images.device)
            
            # BioGPT has a hard 1024 context limit
            current_total_len = prefix_embeds.shape[1] + curr_input_ids.shape[1]
            allowed_to_generate = max(0, 1024 - current_total_len)
            
            for _ in range(min(max_new_tokens, allowed_to_generate)):
                text_embeds = self.llm.biogpt.embed_tokens(curr_input_ids)
                full_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)
                
                outputs = self.llm(inputs_embeds=full_embeds)
                next_token_logits = outputs.logits[:, -1, :]

                for i in range(curr_input_ids.shape[0]):
                    for token_id in set(curr_input_ids[i].tolist()):
                        if next_token_logits[i, token_id] > 0:
                            next_token_logits[i, token_id] /= repetition_penalty
                        else:
                            next_token_logits[i, token_id] *= repetition_penalty

                next_token_logits = next_token_logits / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                finished |= (next_token.squeeze(-1) == tokenizer.eos_token_id)
                curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)
                
                if finished.all():
                    break
                    
            return curr_input_ids