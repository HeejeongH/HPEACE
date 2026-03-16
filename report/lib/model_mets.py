import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=4, num_layers=3, lat_act=False, norm_type='none'):
        super(AutoEncoder, self).__init__()
        
        layers = []
        curr_dim = input_dim

        if num_layers == 3: 
            hidden_dims = [latent_dim * 4, latent_dim * 2, latent_dim] 
        
        elif num_layers == 2: 
            hidden_dims = [latent_dim * 2, latent_dim]
        
        elif num_layers == 4:
            hidden_dims = [latent_dim * 4, latent_dim * 3, latent_dim * 2, latent_dim]
        
        elif num_layers == -1:
            hidden_dims = [latent_dim * 2, latent_dim * 2, latent_dim]

        for h_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(curr_dim, h_dim))
            if norm_type == 'batch':
                layers.append(nn.BatchNorm1d(h_dim))
            elif norm_type == 'layer':
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
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

        # --- [Decoder] (신규 추가) ---
        # Encoder의 역순으로 구성
        dec_layers = []
        curr_dim = latent_dim
        reversed_hidden = hidden_dims[::-1] # [latent, latent*2 ...]
        
        # latent -> hidden layers
        for h_dim in reversed_hidden:
            dec_layers.append(nn.Linear(curr_dim, h_dim))
            dec_layers.append(nn.ReLU())
            curr_dim = h_dim
            
        # 마지막 출력층 (Input Dimension으로 복원)
        dec_layers.append(nn.Linear(curr_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        h = self.encoder_backbone(x)
        latent = self.fc_latent(h)
        logit = self.classifier(latent)
        recon = self.decoder(latent)
        return latent, logit, recon