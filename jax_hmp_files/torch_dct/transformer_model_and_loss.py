import torch
import torch.nn as nn

class FrequencyAwareAttention(nn.Module):
    """
    Custom attention module that weights different frequency components differently.
    Extends standard multi-head attention with learnable frequency importance weights.
    """
    def __init__(self, d_model, nhead):
        """
        Args:
            d_model (int): Size of input feature dimension
            nhead (int): Number of attention heads
        """
        super().__init__()
        # embed_dim: Total dimension of the model.
        # num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
        #     across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        self.mha = nn.MultiheadAttention(d_model, nhead)
        # Learnable weights for each frequency component
        self.freq_weights = nn.Parameter(torch.ones(1, 1, d_model))
        # This adds additional learnable parameter
    def forward(self, x):
        """
        Apply frequency-weighted attention to input sequence.
        
        Args:
            x (tensor): Input sequence [seq_len, batch_size, d_model]
            
        Returns:
            tensor: Attention output with same shape as input
        """
        weighted_x = x * self.freq_weights
        # print("weighted_x:", weighted_x.size())
        
        # attn_output, attn_output_weights = multihead_attn(query, key, value),
        #  so we input same query key value for the mha
        return self.mha(weighted_x, weighted_x, weighted_x)

class DCTPoseTransformerBlock(nn.Module):
    """
    Transformer block with separate processing paths for low and high frequency components.
    Combines frequency-aware attention with frequency-specific feed-forward networks.
    """
    def __init__(self, d_model, nhead, dim_feedforward=1024):
        """
        Args:
            d_model (int): Model dimension
            nhead (int): Number of attention heads
            dim_feedforward (int): Dimension of feed-forward network
        """
        super().__init__()
        self.freq_attn = FrequencyAwareAttention(d_model, nhead)
        
        # Process low frequency components with dedicated network
        self.low_freq_net = nn.Sequential(
            nn.Linear(d_model // 2, dim_feedforward // 2),
            nn.GELU(),
            nn.Linear(dim_feedforward // 2, d_model // 2)
        )
        
        # Process high frequency components with dedicated network
        self.high_freq_net = nn.Sequential(
            nn.Linear(d_model // 2, dim_feedforward // 2),
            nn.GELU(),
            nn.Linear(dim_feedforward // 2, d_model // 2)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Process input through attention and frequency-specific networks.
        
        Args:
            x (tensor): Input features [seq_len, batch_size, d_model]
            
        Returns:
            tensor: Processed features with same shape as input
        """
        # Multi-head attention with residual connection
        norm1_x = self.norm1(x)
        attn_output, _ = self.freq_attn(norm1_x)
        x = x + attn_output
        
        # Split and process frequency components separately
        norm2_x = self.norm2(x)
        low_freq_out = self.low_freq_net(norm2_x[..., :norm2_x.size(-1)//2])
        high_freq_out = self.high_freq_net(norm2_x[..., norm2_x.size(-1)//2:])
        
        # Combine frequency components and add residual
        ff_output = torch.cat([low_freq_out, high_freq_out], dim=-1)
        x = x + ff_output
        
        return x

class UncertaintyEmbedding(nn.Module):
    """
    Processes input uncertainties parallel to main network.
    Learns how much uncertainty information should influence the main prediction.
    """
    def __init__(self, uncertainty_dim, d_model):
        """
        Args:
            uncertainty_dim (int): Dimension of input uncertainty features
            d_model (int): Model dimension to embed into
        """
        super().__init__()
        self.uncertainty_embed = nn.Sequential(
            nn.Linear(uncertainty_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # Learnable scale to control uncertainty influence (initialized to 0)
        self.uncertainty_scale = nn.Parameter(torch.zeros(1))
        
    def forward(self, uncertainty):
        """
        Embed and scale uncertainty features.
        
        Args:
            uncertainty (tensor): Input uncertainty features
            
        Returns:
            tensor: Scaled uncertainty embeddings
        """
        uncertainty_features = self.uncertainty_embed(uncertainty)
        scale = torch.sigmoid(self.uncertainty_scale)  # Bound between 0 and 1
        return uncertainty_features * scale

class UncertaintyHead(nn.Module):
    """
    Predicts pose uncertainties using both pose features and embedded uncertainties.
    Outputs variance parameters and covariance matrix factors for each joint.
    """
    def __init__(self, d_model, seq_len, seq_len_output, num_joints=22, coords_per_joint=3):
        """
        Args:
            d_model (int): Model dimension
            seq_len (int): Input sequence length
            seq_len_output (int): Output sequence length
            num_joints (int): Number of joints to predict
            coords_per_joint (int): Number of coordinates per joint (typically 3 for 3D)
        """
        super().__init__()
        self.seq_len = seq_len
        self.seq_len_output = seq_len_output
        self.num_joints = num_joints
        self.coords_per_joint = coords_per_joint
        self.params_per_joint = coords_per_joint * 2  # Variance + covariance parameters
        
        # MLP for processing pose features
        self.mlp = nn.Sequential(
            nn.Linear(d_model * seq_len, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, seq_len_output * num_joints * self.params_per_joint)
        )
        
        # Network for processing embedded uncertainties
        self.uncertainty_processor = nn.Sequential(
            nn.Linear(d_model * seq_len, 512),
            nn.ReLU(),
            nn.Linear(512, seq_len_output * num_joints * self.params_per_joint)
        )
        
        # Weight for combining pose-based and explicit uncertainties
        self.uncertainty_weight = nn.Parameter(torch.zeros(1))
        
        # Initialize network weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with small random values"""
        with torch.no_grad():
            for module in [self.mlp, self.uncertainty_processor]:
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, std=0.01)
                        nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, features, uncertainty_features=None):
        """
        Predict uncertainty parameters from features and optional explicit uncertainties.
        
        Args:
            features (tensor): Pose features [seq_len, batch_size, d_model]
            uncertainty_features (tensor, optional): Explicit uncertainty features
            
        Returns:
            tuple: (variance parameters, covariance parameters)
        """
        batch_size = features.shape[1]
        flattened = features.transpose(0, 1).reshape(batch_size, -1)
        uncertainty_from_features = self.mlp(flattened)
        
        if uncertainty_features is not None:
            # Process and combine with pose-based uncertainties
            uncertainty_flat = uncertainty_features.transpose(0, 1).reshape(batch_size, -1)
            processed_uncertainty = self.uncertainty_processor(uncertainty_flat)
            
            weight = torch.sigmoid(self.uncertainty_weight)
            uncertainty_params = (1 - weight) * uncertainty_from_features + weight * processed_uncertainty
        else:
            uncertainty_params = uncertainty_from_features
        
        # Reshape and split into variance and covariance parameters
        uncertainty_params = uncertainty_params.view(batch_size, self.seq_len_output, 
                                                   self.num_joints, self.params_per_joint)
        
        var_params = uncertainty_params[..., :self.coords_per_joint]
        cov_params = uncertainty_params[..., self.coords_per_joint:]
        
        return var_params, cov_params

class DCTPoseTransformer(nn.Module):
    """
    Main model for pose prediction with uncertainty estimation.
    Combines frequency-aware transformer with uncertainty prediction.
    """
    def __init__(self, input_dim=39, d_model=128, nhead=4, num_layers=2, seq_len=25, seq_len_output=10):
        """
        Args:
            input_dim (int): Input pose dimension (joints * 3)
            d_model (int): Internal model dimension
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer blocks
            seq_len (int): Input sequence length
            seq_len_output (int): Output sequence length
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # Pose embedding
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # Learnable frequency-based positional encoding
        # Tianle: I do not get why this add a learnable position encoding to preserve temporal information in frequency space,
        self.freq_pos_embed = nn.Parameter(torch.randn(seq_len, 1, d_model))
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            DCTPoseTransformerBlock(d_model, nhead) for _ in range(num_layers)
        ])
        
        # Split dimensions for frequency components
        self.low_freq_features = (input_dim + 1) // 2
        self.high_freq_features = input_dim - self.low_freq_features
        
        # Decoders for frequency components
        self.low_freq_decoder = nn.Linear(self.d_model // 2, self.low_freq_features)
        self.high_freq_decoder = nn.Linear(self.d_model // 2, self.high_freq_features)
        
        # Uncertainty components
        uncertainty_dim = input_dim//3 * 3 * 2  # num_joints * coords_per_joint * (var + cov)
        self.uncertainty_embedding = UncertaintyEmbedding(uncertainty_dim, d_model)
        self.uncertainty_head = UncertaintyHead(d_model, seq_len, seq_len_output, 
                                              num_joints=input_dim//3, coords_per_joint=3)
        
    def forward(self, x, input_uncertainty=None):
        """
        Forward pass through the model.
        
        Args:
            x (tensor): Input pose sequence [batch_size, seq_len, input_dim]
            input_uncertainty (tensor, optional): External uncertainty information
            
        Returns:
            tuple: (predicted poses, (variance parameters, covariance parameters))
        """
        batch_size = x.shape[0]
        
        # Embed and add positional encoding
        x = self.input_embed(x)
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model] from input_dim39 to embeded d_model=128
        # with batch_first= False, The resulting attention outputs and weights are consistent with “time-first.”
        x = x + self.freq_pos_embed
        
        # # Process uncertainty if provided
        # uncertainty_features = None
        # if input_uncertainty is not None:
        #     uncertainty_features = self.uncertainty_embedding(input_uncertainty)
        #     uncertainty_features = uncertainty_features.unsqueeze(0).expand(self.seq_len, -1, -1)
        #     x = x + uncertainty_features
        
        # Store intermediate features
        features = []
        uncertainty_enhanced_features = []
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
            features.append(x.clone())
            # if uncertainty_features is not None:
            #     uncertainty_enhanced_features.append(x.clone())
        
        # Decode poses
        x = x.transpose(0, 1)
        low_freq, high_freq = torch.split(x, x.size(-1) // 2, dim=-1)
        low_freq_out = self.low_freq_decoder(low_freq)
        high_freq_out = self.high_freq_decoder(high_freq)
        poses = torch.cat([low_freq_out, high_freq_out], dim=-1)
        
        # Verify output dimension
        assert poses.shape[-1] == self.input_dim, f"Output dimension {poses.shape[-1]} does not match input dimension {self.input_dim}"
        
        # Predict uncertainties
        with torch.no_grad():
            features_detached = [f.detach() for f in features]
            # if uncertainty_features is not None:
            #     uncertainty_features_detached = uncertainty_features.detach()
            # else:
            #     uncertainty_features_detached = None
        
        var_params, cov_params = self.uncertainty_head(features_detached[-1], None)
                                            #uncertainty_features)
        
        return poses, (var_params, cov_params)

def pose_prediction_loss(pred_poses, target_poses):
    """
    Compute L1 loss between predicted and target poses.
    Uses mean absolute error which is more robust to outliers than MSE.
    
    Args:
        pred_poses (tensor): Predicted poses [batch_size, seq_len, joints*3]
        target_poses (tensor): Ground truth poses [batch_size, seq_len, joints*3]
        
    Returns:
        tensor: Mean absolute error across all dimensions
    """
    return torch.mean(torch.abs(pred_poses - target_poses))

def uncertainty_loss_with_covariance(y_true, y_pred, var_params, cov_params, beta=0.5, lambda_cov=0.01, return_cov_only=False):
    """
    Compute uncertainty loss using Cholesky decomposition for numerical stability.
    Implements a negative log likelihood loss with covariance regularization.
    
    Args:
        y_true (tensor): Ground truth poses [batch_size, seq_len, joints*3]
        y_pred (tensor): Predicted poses [batch_size, seq_len, joints*3]
        var_params (tensor): Log variance parameters [batch_size, seq_len, joints, 3]
        cov_params (tensor): Raw covariance factors [batch_size, seq_len, joints, 3]
        beta (float): Temperature parameter for uncertainty scaling (default: 0.5)
        lambda_cov (float): Weight for covariance regularization (default: 0.01)
        return_cov_only (bool): If True, only return constructed covariance matrix
        
    Returns:
        tuple: (total_loss, covariance_matrix)
            - total_loss: Combined uncertainty loss (None if return_cov_only=True)
            - covariance_matrix: Full 3x3 covariance matrices [batch_size, seq_len, joints, 3, 3]
    """
    # Get dimensions from input
    B, T, JC = y_pred.shape  # batch_size, seq_len, joints*coords
    J = JC // 3  # number of joints
    C = 3  # number of coordinates per joint

    # Reshape inputs to separate joint and coordinate dimensions
    y_true = y_true.view(B, T, J, C)
    y_pred = y_pred.view(B, T, J, C)
    var_params = var_params.view(B, T, J, C)
    cov_params = cov_params.view(B, T, J, C)

    # Convert log variance parameters to actual variances (in meters squared)
    variance = torch.exp(var_params)  # Variances in m^2
    var_x, var_y, var_z = variance[..., 0], variance[..., 1], variance[..., 2]

    # Construct Cholesky factor (L) of covariance matrix
    # Covariance = L * L^T ensures positive definiteness
    L = torch.zeros(B, T, J, C, C, device=y_pred.device)
    # Build lower triangular Cholesky factor
    # Scale back to millimeters
    L[..., 0, 0] = torch.sqrt(var_x) * 1000  # x variance
    L[..., 1, 0] = cov_params[..., 0] * torch.sqrt(var_x * var_y) * 1000  # xy covariance
    L[..., 1, 1] = torch.sqrt(var_y) * 1000  # y variance
    L[..., 2, 0] = cov_params[..., 1] * torch.sqrt(var_x * var_z) * 1000  # xz covariance
    L[..., 2, 1] = cov_params[..., 2] * torch.sqrt(var_y * var_z) * 1000  # yz covariance
    L[..., 2, 2] = torch.sqrt(var_z) * 1000  # z variance

    # Compute full covariance matrix from Cholesky factors
    cov_matrix = torch.matmul(L, L.transpose(-1, -2))  # [B, T, J, 3, 3]

    if return_cov_only:
        return None, cov_matrix
    
    # Compute prediction errors (detached to prevent gradients through predictions)
    with torch.no_grad():
        diff = y_true - y_pred.detach()

    # Reshape for batch matrix operations
    L = L.view(B * T * J, C, C)
    diff = diff.view(B * T * J, C)

    # Compute log determinant of covariance (using Cholesky factor)
    diag_L = torch.diagonal(L, dim1=-2, dim2=-1)  # Get diagonal elements
    log_det = 2 * torch.sum(torch.log(diag_L), dim=1)  # Sum logs for determinant

    # Compute Mahalanobis distance (normalized error under covariance)
    # Solve L * x = diff for x, then compute x^T * x
    m = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False)
    mahalanobis = torch.sum(m ** 2, dim=1).squeeze()

    # Combine into negative log likelihood
    nll = 0.5 * (log_det + mahalanobis)
    nll = nll.view(B, T, J)

    # Add regularization terms (optional based on lambda parameters)
    reg_cov = lambda_cov * torch.mean(torch.abs(cov_params))  # L1 regularization on covariances
    lambda_var = 1e-3  # Small weight for variance regularization
    reg_var = lambda_var * torch.mean(torch.exp(-var_params))  # Prevent too-small variances

    # Compute final loss (mean over all dimensions)
    loss = torch.mean(nll)  # + reg_var + reg_cov  # Uncomment to add regularization

    return loss, cov_matrix
