import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import GPT2Model

from .base_model import BaseModel
from utils import masked_mae


class FocalLoss(nn.Module):
    """Focal Loss to down-weight easy examples for class imbalance."""

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None


class GRL(nn.Module):
    def __init__(self, coeff=1.0):
        super().__init__()
        self.coeff = coeff

    def forward(self, x):
        return GradientReversalFn.apply(x, self.coeff)


class DebiasLoRABlock(nn.Module):
    """LoRA block producing the residual delta."""

    def __init__(self, d: int, m: int, dropout: float = 0.0):
        super().__init__()
        self.A = nn.Linear(d, m, bias=False)
        self.B = nn.Linear(m, d, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, r):
        return self.B(self.A(self.dropout(r)))


class GPT2WithDebiasLoRA(nn.Module):
    def __init__(self, selected_layers_for_lora, lora_rank=16, lora_dropout=0.1):
        super().__init__()
        self.selected_layers_for_lora = set(selected_layers_for_lora)
        self.enable_lora = False

        self.gpt2 = GPT2Model.from_pretrained(
            "gpt2", output_attentions=False, output_hidden_states=True
        )
        for param in self.gpt2.parameters():
            param.requires_grad = False

        self.config = self.gpt2.config
        self.d = self.config.hidden_size
        self.lora_rank = lora_rank
        self.lora_dropout = lora_dropout

        self.debias_loras = nn.ModuleDict()

    def init_lora_layers(self, device):
        """Initialize LoRA blocks for selected layers if missing."""
        for layer_idx in self.selected_layers_for_lora:
            key = str(layer_idx)
            if key in self.debias_loras:
                continue
            self.debias_loras[key] = DebiasLoRABlock(
                d=self.d, m=self.lora_rank, dropout=self.lora_dropout
            ).to(device)

    def forward(self, inputs_embeds):
        hidden_states = inputs_embeds
        batch_size, seq_len, _ = hidden_states.shape
        device = inputs_embeds.device

        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_embeds = self.gpt2.wpe(position_ids)
        hidden_states = hidden_states + position_embeds
        hidden_states = self.gpt2.drop(hidden_states)

        all_hidden_states = ()

        for i, block in enumerate(self.gpt2.h):
            outputs = block(hidden_states)
            hidden_states = outputs[0]

            if self.enable_lora and str(i) in self.debias_loras:
                delta = self.debias_loras[str(i)](hidden_states)
                hidden_states = hidden_states + delta

            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.gpt2.ln_f(hidden_states)

        return {"last_hidden_state": hidden_states, "hidden_states": all_hidden_states}


class DomainProbe(nn.Module):
    """Probe used in adversarial debiasing."""

    def __init__(self, in_dim=768, num_classes=9, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)


class CallST(BaseModel):
    """
    GPT-2 backbone with LoRA-based debiasing.

    The forward path always returns predictions shaped as (B, N, 1, horizon) so it
    plugs into the existing Lightning training/validation flow.
    """

    def __init__(self, config, data_module):
        super().__init__(config)

        model_config = config["model"]
        self.stage = model_config.get("stage", "pretrain")
        self.top_k = model_config.get("top_k", 3)
        self.seq_len = data_module.num_timesteps_input
        self.horizon = data_module.num_timesteps_output

        self.gpt_channel = model_config.get("hidden_channels", 256)
        self.to_gpt_channel = 768

        # Embedding layers
        self.start_conv_speed = nn.Conv2d(
            self.seq_len, self.gpt_channel, kernel_size=(1, 1)
        )
        self.start_conv_flow = nn.Conv2d(
            self.seq_len, self.gpt_channel, kernel_size=(1, 1)
        )
        self.feature_fusion = nn.Conv2d(
            self.gpt_channel * 3, self.to_gpt_channel, kernel_size=(1, 1)
        )

        expected_nodes = self._infer_num_nodes(data_module)
        self.num_types, type_ids = self._init_road_type_ids(expected_nodes)
        self.register_buffer("type_ids", type_ids, persistent=False)
        self.road_type_embedding = nn.Embedding(self.num_types, self.gpt_channel)

        self.num_incident = 9
        self.label_embedding = nn.Embedding(self.num_incident, self.to_gpt_channel)

        self.gpt2_lora = GPT2WithDebiasLoRA(
            selected_layers_for_lora=[], lora_rank=16
        )
        self.num_layers = len(self.gpt2_lora.gpt2.h)

        default_layers = list(range(min(self.top_k, self.num_layers)))
        selected_layers = model_config.get("selected_layers", default_layers)
        self.set_selected_layers(selected_layers, init_lora=False)

        self.probes = nn.ModuleList(
            [nn.Linear(self.to_gpt_channel, self.num_incident) for _ in range(self.num_layers)]
        )

        self.debias_coeff = 1.0
        self.grl = GRL(coeff=self.debias_coeff)
        self.domain_probe = DomainProbe(
            in_dim=self.to_gpt_channel, num_classes=self.num_incident
        )

        self.regression_layer = nn.Conv2d(
            self.to_gpt_channel * 2, self.horizon, kernel_size=(1, 1)
        )

        class_counts = model_config.get("class_counts")
        if class_counts is not None:
            counts = torch.tensor(class_counts, dtype=torch.float32)
            weights = 1.0 / (counts + 1.0)
            weights = weights / weights.sum() * len(counts)
            self.class_weights = weights
        else:
            self.class_weights = None

        self.loss_fct = FocalLoss(alpha=self.class_weights, gamma=2.0)

    def _infer_num_nodes(self, data_module):
        if not getattr(data_module, "dataset_configs", None):
            return None
        adj_path = data_module.dataset_configs[0].get("adj_path")
        if adj_path and os.path.exists(adj_path):
            adj = np.load(adj_path)
            return adj.shape[0]
        return None

    def _init_road_type_ids(self, expected_nodes):
        road_type_path = self.config.get("model", {}).get(
            "road_type_path",
            os.path.join(self.config.get("data", {}).get("root_dir", "./dataset"), "sacramento_sensor_type.csv"),
        )

        if expected_nodes is None:
            expected_nodes = 1

        if os.path.exists(road_type_path):
            road_df = pd.read_csv(road_type_path, index_col=0)
            type_series = road_df["Type"]
            unique_types = sorted(type_series.unique().tolist())
            type2id = {t: i for i, t in enumerate(unique_types)}

            type_ids_np = np.zeros(expected_nodes, dtype=np.int64)

            if "node_index" in road_df.columns:
                for _, row in road_df.iterrows():
                    idx = int(row["node_index"])
                    if 0 <= idx < expected_nodes:
                        type_ids_np[idx] = type2id[row["Type"]]
            else:
                mapped = type_series.map(type2id).values
                limit = min(expected_nodes, len(mapped))
                type_ids_np[:limit] = mapped[:limit]

            num_types = len(unique_types)
        else:
            num_types = 1
            type_ids_np = np.zeros(expected_nodes, dtype=np.int64)

        return num_types, torch.tensor(type_ids_np, dtype=torch.long)

    def _reshape_input(self, x):
        # x can be (B, N, 1, T) or (B, F, 1, N, T)
        if x.dim() == 5:
            x = x.squeeze(2)  # (B, F, N, T)
            x = x.permute(0, 3, 2, 1)  # (B, T, N, F)
        elif x.dim() == 4:
            x = x.permute(0, 3, 1, 2)  # (B, T, N, 1)
        else:
            raise ValueError(f"Unexpected input shape: {tuple(x.shape)}")
        return x

    def _build_gpt_inputs(self, input):
        input = self._reshape_input(input)
        batch_size, _, num_nodes, num_features = input.shape

        flow = input[..., 0]
        speed = input[..., 2] if num_features > 2 else flow
        incident = input[..., 3] if num_features > 3 else torch.zeros_like(flow)

        x_enc_flow = flow.permute(0, 2, 1).unsqueeze(-1)
        x_enc_flow = x_enc_flow.permute(0, 2, 1, 3)
        x_enc_flow = self.start_conv_flow(x_enc_flow)

        x_enc_speed = speed.permute(0, 2, 1).unsqueeze(-1)
        x_enc_speed = x_enc_speed.permute(0, 2, 1, 3)
        x_enc_speed = self.start_conv_speed(x_enc_speed)

        type_ids = self.type_ids
        if type_ids.numel() != num_nodes:
            type_ids = torch.arange(num_nodes, device=input.device) % self.num_types
        type_emb = self.road_type_embedding(type_ids.to(input.device))
        type_emb = type_emb.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, -1)
        type_emb = type_emb.permute(0, 2, 1, 3)

        data_st = torch.cat([x_enc_flow, x_enc_speed, type_emb], dim=1)
        data_st = self.feature_fusion(data_st)

        gpt_inputs = data_st.squeeze(-1).permute(0, 2, 1)

        labels = incident.max(dim=1).values.long()

        return gpt_inputs, labels

    def _format_target(self, y):
        if y.dim() == 5:
            y = y[:, 0, :, :, :]
            y = y.permute(0, 2, 1, 3)
        return y

    def forward(self, input, adj=None, stage=None, return_aux=False):
        stage = stage or self.stage
        gpt_inputs, labels = self._build_gpt_inputs(input)
        labels = labels.to(gpt_inputs.device)
        label_emb_vector = self.label_embedding(labels)

        if stage == "pretrain":
            self.gpt2_lora.enable_lora = False
            outputs = self.gpt2_lora(inputs_embeds=gpt_inputs)
            last_hidden = outputs["last_hidden_state"]

            combined = torch.cat([last_hidden, label_emb_vector], dim=-1)
            combined = combined.permute(0, 2, 1).unsqueeze(-1)

            prediction = self.regression_layer(combined)
            return prediction.permute(0, 2, 3, 1).contiguous()

        if stage == "probe":
            self.gpt2_lora.enable_lora = False

            with torch.no_grad():
                outputs = self.gpt2_lora(inputs_embeds=gpt_inputs)
                hidden_states = outputs["hidden_states"]

            layer_losses = []
            layer_f1_scores = []

            labels_cpu = labels.view(-1).cpu().numpy()

            for l in range(self.num_layers):
                h_l = hidden_states[l]
                logits = self.probes[l](h_l.detach())

                loss = self.loss_fct(
                    logits.view(-1, self.num_incident), labels.view(-1)
                )
                layer_losses.append(loss)

                preds = logits.argmax(dim=-1).view(-1).cpu().numpy()
                f1 = f1_score(labels_cpu, preds, average="macro", zero_division=0)
                layer_f1_scores.append(f1)

            avg_probe_loss = sum(layer_losses) / len(layer_losses)
            return avg_probe_loss, layer_f1_scores

        if stage == "task":
            self._ensure_lora_initialized(gpt_inputs.device)
            self.gpt2_lora.enable_lora = True

            outputs = self.gpt2_lora(inputs_embeds=gpt_inputs)
            hidden_states = outputs["hidden_states"]

            adv_losses = []

            for l_idx in self.gpt2_lora.selected_layers_for_lora:
                h_l = hidden_states[int(l_idx)]
                h_flat = h_l.reshape(-1, self.to_gpt_channel)
                h_rev = self.grl(h_flat)
                domain_logits = self.domain_probe(h_rev)

                loss = self.loss_fct(domain_logits, labels.view(-1))
                adv_losses.append(loss)

            loss_debias = sum(adv_losses) / (len(adv_losses) + 1e-8) if adv_losses else torch.tensor(0.0, device=gpt_inputs.device)

            last_hidden = outputs["last_hidden_state"]
            combined = torch.cat([last_hidden, label_emb_vector], dim=-1)
            combined = combined.permute(0, 2, 1).unsqueeze(-1)

            prediction = self.regression_layer(combined)
            prediction = prediction.permute(0, 2, 3, 1).contiguous()

            if return_aux:
                return prediction, loss_debias
            return prediction

        raise ValueError(f"Unknown stage: {stage}")

    def set_selected_layers(self, layer_indices, init_lora=True):
        self.gpt2_lora.selected_layers_for_lora = set(map(int, layer_indices))
        if init_lora:
            self._ensure_lora_initialized(device=self.device)

    def _ensure_lora_initialized(self, device):
        if not self.gpt2_lora.debias_loras:
            self.gpt2_lora.init_lora_layers(device)

    def training_step(self, batch, batch_idx):
        x, y, adj = batch[:3]
        current_batch_size = x.size(0)
        stage = self.stage
        y_formatted = self._format_target(y)

        if stage == "probe":
            loss, f1_scores = self.forward(x, adj, stage=stage)
            self.log(
                "probe_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
                batch_size=current_batch_size,
            )
            self.log(
                "probe_f1_mean",
                torch.tensor(f1_scores, device=loss.device).mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
                batch_size=current_batch_size,
            )
            return loss

        if stage == "task":
            preds, debias_loss = self.forward(x, adj, stage=stage, return_aux=True)
            main_loss = masked_mae(preds, y_formatted)
            total_loss = main_loss + debias_loss
            self.log(
                "train_loss",
                total_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
                batch_size=current_batch_size,
            )
            self.log(
                "train_main_loss",
                main_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=current_batch_size,
            )
            self.log(
                "train_debias_loss",
                debias_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=current_batch_size,
            )
            return total_loss

        preds = self.forward(x, adj, stage=stage)
        loss = masked_mae(preds, y_formatted)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=current_batch_size,
        )
        return loss
