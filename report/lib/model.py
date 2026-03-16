import numpy as np
import torch
import torch.nn as nn


class EmbeddingAutoEncoder(nn.Module):
    def __init__(self, cardinalities, disease_short_name='IWC', emb_dropout=0.1):
        super(EmbeddingAutoEncoder, self).__init__()
        
        self.embeddings = nn.ModuleList()
        self.total_emb_dim = 0

        for n_unique in cardinalities:
            emb_dim = min(32, max(4, int(np.log2(1 + n_unique) * 2)))
            self.embeddings.append(nn.Embedding(num_embeddings=n_unique, embedding_dim=emb_dim))
            self.total_emb_dim += emb_dim
            
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        if disease_short_name != "IWC":
            raise ValueError
        
        lat_act = True
        hidden_dim = 16
        latent_dim = 8
        hidden_dims = [hidden_dim, hidden_dim // 4, latent_dim]
        norm_type = 'none'

        layers = []
        curr_dim = self.total_emb_dim

        for h_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(curr_dim, h_dim))
            if norm_type == 'batch':
                layers.append(nn.BatchNorm1d(h_dim))
            elif norm_type == 'layer':
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.3))
            curr_dim = h_dim
            
        self.encoder_backbone = nn.Sequential(*layers)

        lat_layers = []
        lat_layers.append(nn.Linear(curr_dim, latent_dim))
        if lat_act:
            lat_layers.append(nn.Tanh())
        self.fc_latent = nn.Sequential(*lat_layers)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, 1)
        )

        dec_layers = []
        curr_dim = latent_dim
        reversed_hidden = hidden_dims[::-1] 
        
        for h_dim in reversed_hidden:
            dec_layers.append(nn.Linear(curr_dim, h_dim))
            dec_layers.append(nn.ReLU())
            curr_dim = h_dim
            
        # Output dimension = Total Embedding Dimension
        dec_layers.append(nn.Linear(curr_dim, self.total_emb_dim)) 
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        
        emb_outputs = []
        for i, emb_layer in enumerate(self.embeddings):
            col_input = x[:, i].long() 
            emb_outputs.append(emb_layer(col_input))
            
        x_embedded = torch.cat(emb_outputs, dim=1)
        x_embedded = self.emb_dropout(x_embedded)
        
        h = self.encoder_backbone(x_embedded)
        latent = self.fc_latent(h)
        
        logit = self.classifier(latent)
        recon = self.decoder(latent)
        
        return latent, logit, recon, x_embedded

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, disease_short_name='EBP'):
        super(AutoEncoder, self).__init__()
        
        layers = []
        curr_dim = input_dim
        
        # Hyperparams 설정
        if disease_short_name == "EBP":
            lat_act = False
            hidden_dim = 16
            latent_dim = 16
            hidden_dims = [hidden_dim * 4, hidden_dim // 4, latent_dim]
            norm_type = 'none'

        elif disease_short_name == "IFG":
            lat_act = False
            hidden_dim = 8
            latent_dim = 128
            hidden_dims = [hidden_dim // 2, hidden_dim * 2, hidden_dim // 2, latent_dim]
            norm_type = 'layer'
        
        elif disease_short_name == "ET":
            lat_act = True
            hidden_dim = 8
            latent_dim = 4
            hidden_dims = [hidden_dim // 2, hidden_dim * 2, hidden_dim // 2, latent_dim]
            norm_type = 'none'

        elif disease_short_name == "DHDL-C":
            lat_act = True
            hidden_dim = 8
            latent_dim = 16
            hidden_dims = [hidden_dim, hidden_dim // 2, latent_dim]
            norm_type = 'layer'
       
        else:
            raise ValueError

        for h_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(curr_dim, h_dim))
            if norm_type == 'batch':
                layers.append(nn.BatchNorm1d(h_dim))
            elif norm_type == 'layer':
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.3))
            curr_dim = h_dim
            
        self.encoder_backbone = nn.Sequential(*layers)

        lat_layers = []
        lat_layers.append(nn.Linear(curr_dim, latent_dim))
        if lat_act:
            lat_layers.append(nn.Tanh())
        self.fc_latent = nn.Sequential(*lat_layers)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, 1)
        )

        # --- [Decoder] ---
        dec_layers = []
        curr_dim = latent_dim
        reversed_hidden = hidden_dims[::-1] # [latent, latent*2 ...]
        
        for h_dim in reversed_hidden:
            dec_layers.append(nn.Linear(curr_dim, h_dim))
            dec_layers.append(nn.ReLU())
            curr_dim = h_dim
            
        dec_layers.append(nn.Linear(curr_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        h = self.encoder_backbone(x)
        latent = self.fc_latent(h)
        logit = self.classifier(latent)
        recon = self.decoder(latent)
        return latent, logit, recon