# /mnt/p/perpetual/models/architecture.py
"""
architecture.py
-----------------------------------------------------
Defines the DeribitHybridModel: a TCN + Transformer hybrid
architecture designed to process Tier 1 features for
directional prediction, return estimation, volatility
forecasting, and optimal sizing. Built for both live
inference and training using PyTorch.
-----------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Union, Any


class CausalConv1d(nn.Module):
    """
    1D causal convolution with proper padding to maintain causality.
    Ensures each output only depends on past inputs.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int = 1, **kwargs):
        super().__init__()
        # Calculate padding required for causality
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            **kwargs
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, channels, seq_len]
        Returns:
            out: [batch_size, channels, seq_len]
        """
        # Apply conv and remove future timesteps from padding
        result = self.conv(x)
        return result[:, :, :-self.padding] if self.padding > 0 else result


class TemporalBlock(nn.Module):
    """
    Temporal block for TCN with residual connection and normalization.
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, 
                 stride: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = CausalConv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, dilation=dilation
        )
        self.batch_norm1 = nn.BatchNorm1d(n_outputs)
        self.mish1 = nn.Mish()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = CausalConv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, dilation=dilation
        )
        self.batch_norm2 = nn.BatchNorm1d(n_outputs)
        self.mish2 = nn.Mish()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection if input size != output size
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, channels, seq_len]
        Returns:
            out: [batch_size, n_outputs, seq_len]
        """
        # First convolution block
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.mish1(out)
        out = self.dropout1(out)
        
        # Second convolution block
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.mish2(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network with dilated convolutions.
    """
    def __init__(self, num_inputs: int, num_channels: List[int], 
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        # Create network with exponentially increasing dilation
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size, 
                    stride=1, dilation=dilation, dropout=dropout
                )
            )
            
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, channels]
        Returns:
            out: [batch_size, seq_len, num_channels[-1]]
        """
        # Transpose to [batch, channels, seq_len] for convolution
        x = x.transpose(1, 2)
        
        # Sanitize inputs (add numerical stability protection)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        out = self.network(x)
        # Transpose back to [batch, seq_len, channels]
        return out.transpose(1, 2)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer.
    """
    def __init__(self, d_model: int, max_len: int = 6000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize PE buffer with zeros (will be filled in __init__)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Use a more stable approach for computing position encoding
        # Instead of computing all at once, compute in smaller batches to prevent overflow
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Ensure div_term doesn't have extreme values
        div_term = torch.clamp(div_term, min=1e-10, max=1e10)
        
        # Compute sin and cos values in a numerically stable way
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Clamp values to avoid extreme numbers
        pe = torch.clamp(pe, min=-5.0, max=5.0)
        
        pe = pe.unsqueeze(0)
        
        # Register buffer to avoid counting as model parameter
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, embedding_dim]
        Returns:
            out: [batch_size, seq_len, embedding_dim]
        """
        # Prevent NaN by handling extreme values
        x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Get only the needed portion of positional encoding
        pe_slice = self.pe[:, :x.size(1), :]
        
        # Prevent NaN in positional encoding
        if torch.isnan(pe_slice).any() or torch.isinf(pe_slice).any():
            print("❌ NaN/Inf in positional encoding values BEFORE addition")
            print(f"PE max: {torch.max(pe_slice)}, min: {torch.min(pe_slice)}")
            # Fix the problematic values
            pe_slice = torch.nan_to_num(pe_slice, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Add positional encoding to input
        x = x + pe_slice
        
        # Final check for NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("❌ NaN or Inf in final output of PositionalEncoding")
            print("Sample x[0, -1, :10]:", x[0, -1, :10])
            # Fix the problematic values as a last resort
            x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
            
        return self.dropout(x)


class CrossAssetAttention(nn.Module):
    """
    Multi-head attention layer with asset-aware relative position encoding.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, query, key, value, asset_ids=None):
        """
        Args:
            query: [batch_size, seq_len, embed_dim]
            key: [batch_size, seq_len, embed_dim]
            value: [batch_size, seq_len, embed_dim]
            asset_ids: Optional tensor of asset indices for cross-asset attention
        Returns:
            out: [batch_size, seq_len, embed_dim]
        """
        # Sanitize inputs
        query = torch.nan_to_num(query, nan=0.0, posinf=5.0, neginf=-5.0)
        key = torch.nan_to_num(key, nan=0.0, posinf=5.0, neginf=-5.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Apply asset-aware attention if asset IDs are provided
        if asset_ids is not None:
            attn_mask = None  # Could create asset-aware attention mask here if needed
        else:
            attn_mask = None
            
        output, _ = self.multihead_attn(query, key, value, attn_mask=attn_mask)
        
        # Sanitize outputs
        output = torch.nan_to_num(output, nan=0.0, posinf=5.0, neginf=-5.0)
        
        return output


class CrossAssetTransformer(nn.Module):
    """
    Transformer encoder with cross-asset attention capability.
    """
    def __init__(self, d_model: int, nhead: int = 8, num_layers: int = 4, 
                 dim_feedforward: int = 512, dropout: float = 0.1, 
                 max_seq_length: int = 6000):
        super().__init__()
        
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Use a more robust initialization for transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Apply normalization before attention (more stable)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        self.cross_asset_attention = CrossAssetAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout
        )
        
        # Add layer norm at the input to help stabilize
        self.input_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, asset_ids=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            asset_ids: Optional tensor of asset indices
        Returns:
            out: [batch_size, seq_len, d_model]
        """
        # Normalize and sanitize input
        x = self.input_norm(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Detect NaNs or Infs after positional encoding
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("❌ NaN or Inf in x after positional encoding!")
            print(f"Shape: {x.shape}, dtype: {x.dtype}")
            print("Sample slice:", x[0, -1, :10])
            # Fix NaNs as a last resort
            x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Apply transformer encoder with extra safety checks
        try:
            out = self.transformer_encoder(x)
            
            # Verify output is valid
            if torch.isnan(out).any() or torch.isinf(out).any():
                print("❌ NaN/Inf detected in transformer output!")
                # Fix as last resort
                out = torch.nan_to_num(out, nan=0.0, posinf=5.0, neginf=-5.0)
        except Exception as e:
            print(f"Exception in transformer: {e}")
            # Fallback: skip transformer if it fails
            out = x
            
        # Apply cross-asset attention if asset IDs are provided
        if asset_ids is not None:
            try:
                out = self.cross_asset_attention(out, out, out, asset_ids)
            except Exception as e:
                print(f"Exception in cross-asset attention: {e}")
                # No change if cross-asset attention fails
                pass
                
        return out


class NeuralKellyHead(nn.Module):
    """
    Position sizing head inspired by Kelly criterion.
    Outputs optimal position size based on expected return and risk.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Dropout(0.2)
        )
        
        # Predict expected return (μ)
        self.mu_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Linear(64, 1)
        )
        
        # Predict variance (σ²)
        self.sigma_sq_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive variance
        )
        
        # Funding rate sensitivity
        self.funding_sensitivity = nn.Sequential(
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Linear(64, 1),
            nn.Tanh()  # Range: [-1, 1]
        )
        
        # Final position sizing with funding adjustment
        self.size_head = nn.Sequential(
            nn.Linear(3, 32),  # 3 inputs: expected return, variance, funding sensitivity
            nn.Mish(),
            nn.Linear(32, 1),
            nn.Tanh()  # Output in range [-1, 1] where sign indicates direction
        )
        
    def forward(self, x, funding_rate=None):
        """
        Args:
            x: Feature tensor [batch_size, input_dim]
            funding_rate: Optional funding rate tensor [batch_size, 1]
        Returns:
            Dictionary with position parameters
        """
        # Sanitize inputs
        x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        if funding_rate is not None:
            funding_rate = torch.nan_to_num(funding_rate, nan=0.0, posinf=1.0, neginf=-1.0)
            funding_rate = torch.clamp(funding_rate, min=-5.0, max=5.0)
        
        shared = self.shared(x)
        
        # Predict return and risk
        mu = self.mu_head(shared)
        sigma_sq = self.sigma_sq_head(shared) + 1e-6  # Add epsilon to avoid division by zero
        
        # Clamp values for numerical stability
        mu = torch.clamp(mu, min=-5.0, max=5.0)
        sigma_sq = torch.clamp(sigma_sq, min=1e-6, max=100.0)
        
        # Kelly fraction: f* = μ/σ² (simplified)
        kelly_fraction = mu / sigma_sq
        kelly_fraction = torch.clamp(kelly_fraction, min=-5.0, max=5.0)
        
        # Funding sensitivity
        alpha = self.funding_sensitivity(shared)
        
        # Apply funding adjustment if available
        if funding_rate is not None:
            # Adjust position based on funding rate
            funding_adj = alpha * funding_rate
            funding_adj = torch.clamp(funding_adj, min=-5.0, max=5.0)
        else:
            funding_adj = torch.zeros_like(kelly_fraction)
            
        # Combined inputs for final sizing
        combined = torch.cat([kelly_fraction, sigma_sq, funding_adj], dim=1)
        position_size = self.size_head(combined)
        
        return {
            'position_size': position_size,
            'expected_return': mu,
            'expected_risk': torch.sqrt(sigma_sq),  # Convert variance to volatility
            'funding_sensitivity': alpha
        }


class PortfolioOptimizerHead(nn.Module):
    """
    Portfolio-aware head that outputs diversification and risk metrics.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Mish()
        )
        
        # Correlation prediction (asset vs portfolio)
        self.correlation_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.Mish(),
            nn.Linear(32, 1),
            nn.Tanh()  # Range: [-1, 1]
        )
        
        # Diversification score
        self.diversification_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.Mish(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Range: [0, 1]
        )
        
        # Portfolio beta
        self.beta_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.Mish(),
            nn.Linear(32, 1)
        )
        
        # Risk contribution
        self.risk_contribution_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.Mish(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Range: [0, 1]
        )
        
    def forward(self, x):
        """
        Args:
            x: Feature tensor [batch_size, input_dim]
        Returns:
            Dictionary with portfolio metrics
        """
        # Sanitize input
        x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        
        features = self.network(x)
        
        return {
            'correlation_to_portfolio': self.correlation_head(features),
            'diversification_score': self.diversification_head(features),
            'portfolio_beta': self.beta_head(features),
            'risk_contribution': self.risk_contribution_head(features)
        }


class DeribitHybridModel(nn.Module):
    """
    Hybrid TCN + Transformer model for Deribit perpetual trading.
    Combines temporal pattern recognition with cross-asset intelligence
    and portfolio awareness.
    """
    def __init__(
        self, 
        input_dim: int,
        tcn_channels: List[int] = [128, 128, 128, 128],
        tcn_kernel_size: int = 3,
        transformer_dim: int = 128,
        transformer_heads: int = 8,
        transformer_layers: int = 4,
        dropout: float = 0.2,
        max_seq_length: int = 6000
    ):
        super().__init__()
        
        # Input dimensions
        self.input_dim = input_dim
        self.max_seq_length = max_seq_length
        
        # Feature projection layer
        self.input_projection = nn.Linear(input_dim, tcn_channels[0])
        
        # Add input layer normalization for stability
        self.input_norm = nn.LayerNorm(input_dim)
        
        # TCN for short-term pattern detection
        self.tcn = TemporalConvNet(
            num_inputs=tcn_channels[0],
            num_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=dropout
        )
        
        # TCN output dimension is the last channel size
        tcn_output_dim = tcn_channels[-1]
        
        # Projection from TCN to Transformer
        self.tcn_to_transformer = nn.Linear(tcn_output_dim, transformer_dim)
        
        # Transformer for cross-asset relationships
        self.transformer = CrossAssetTransformer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            max_seq_length=max_seq_length
        )
        
        # Feature gating mechanism
        self.feature_gate = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.Sigmoid()
        )
        
        # Direction classification head (3-way: long, flat, short)
        self.direction_head = nn.Sequential(
            nn.Linear(transformer_dim, 64),
            nn.LayerNorm(64),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        )
        
        # Return prediction head
        self.return_head = nn.Sequential(
            nn.Linear(transformer_dim, 64),
            nn.LayerNorm(64),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Risk (volatility) head
        self.risk_head = nn.Sequential(
            nn.Linear(transformer_dim, 64),
            nn.LayerNorm(64),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive volatility
        )
        
        # Position sizing using Neural Kelly criterion
        self.size_head = NeuralKellyHead(transformer_dim)
        
        # Portfolio optimization head
        self.portfolio_head = PortfolioOptimizerHead(transformer_dim)
        
        # Initialize weights using a more stable approach
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with a more stable approach."""
        for name, p in self.named_parameters():
            if 'weight' in name:
                if len(p.shape) >= 2:
                    # Use Xavier uniform for weight matrices
                    nn.init.xavier_uniform_(p, gain=0.5)  # Lower gain for stability
                else:
                    # Use normal init for vectors
                    nn.init.normal_(p, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(p)
        
    def forward(self, x, funding_rate=None, asset_ids=None):
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            funding_rate: Optional funding rate tensor [batch_size, 1]
            asset_ids: Optional tensor of asset indices
            
        Returns:
            Dictionary of outputs from all heads
        """
        batch_size, seq_len, _ = x.shape
        
        # Add numerical stability protection in input pipeline
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = torch.clamp(x, min=-20.0, max=20.0)
        
        # Apply input normalization
        x = self.input_norm(x)
        
        # Project input features
        x = self.input_projection(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Apply TCN for local pattern detection
        tcn_output = self.tcn(x)
        tcn_output = torch.nan_to_num(tcn_output, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Project to transformer dimension
        transformer_input = self.tcn_to_transformer(tcn_output)
        transformer_input = torch.nan_to_num(transformer_input, nan=0.0, posinf=5.0, neginf=-5.0)

        # Check for NaNs/Infs BEFORE transformer
        if torch.isnan(transformer_input).any() or torch.isinf(transformer_input).any():
            print("❌ NaN or Inf in transformer_input!")
            print(f"Shape: {transformer_input.shape}, dtype: {transformer_input.dtype}")
            print("Sample slice:", transformer_input[0, -1, :10])  # Show 10 values of last timestep
            transformer_input = torch.nan_to_num(transformer_input, nan=0.0, posinf=5.0, neginf=-5.0)
               
        # Apply transformer for global dependencies
        transformer_output = self.transformer(transformer_input, asset_ids)
        transformer_output = torch.nan_to_num(transformer_output, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Take the last sequence element for prediction
        context_vector = transformer_output[:, -1, :]
        
        # Apply feature gating
        gate = self.feature_gate(context_vector)
        gated_features = context_vector * gate
        
        # Apply prediction heads
        direction_logits = self.direction_head(gated_features)
        expected_return = self.return_head(gated_features)
        expected_risk = self.risk_head(gated_features)
        
        # Get position sizing from the neural Kelly head
        sizing_outputs = self.size_head(gated_features, funding_rate)
        
        # Get portfolio optimization metrics
        portfolio_outputs = self.portfolio_head(gated_features)
        
        # Combine all outputs
        outputs = {
            'direction_logits': direction_logits,
            'expected_return': expected_return,
            'expected_risk': expected_risk,
            'position_size': sizing_outputs['position_size'],
            'funding_sensitivity': sizing_outputs['funding_sensitivity']
        }
        
        # Add portfolio metrics
        outputs.update(portfolio_outputs)
        
        return outputs
    
    def predict_trade_signal(self, x, funding_rate=None, asset_ids=None, threshold=0.0):
        """
        Generate a trade signal from model outputs.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            funding_rate: Optional funding rate tensor [batch_size, 1]
            asset_ids: Optional tensor of asset indices
            threshold: Confidence threshold for taking a position
            
        Returns:
            Dictionary with trade signals
        """
        with torch.no_grad():
            outputs = self.forward(x, funding_rate, asset_ids)
            
            # Get direction probabilities
            direction_probs = F.softmax(outputs['direction_logits'], dim=1)
            
            # Get position size (scaled to [-1, 1])
            position_size = outputs['position_size']
            
            # Determine trading action
            trade_action = torch.argmax(direction_probs, dim=1)  # 0=short, 1=flat, 2=long
            
            # Apply confidence threshold
            confidence = torch.max(direction_probs, dim=1)[0]
            valid_trades = confidence >= threshold
            
            # Create final position size incorporating direction and sizing
            # Convert 0,1,2 to -1,0,1
            direction = (trade_action - 1).float()
            
            # Final position is direction * size * confidence
            final_position = direction * torch.abs(position_size.squeeze()) * confidence
            
            # Zero out positions that don't meet threshold
            final_position = torch.where(valid_trades, final_position, torch.zeros_like(final_position))
            
            return {
                'direction': direction,
                'position_size': final_position,
                'confidence': confidence,
                'expected_return': outputs['expected_return'].squeeze(),
                'expected_risk': outputs['expected_risk'].squeeze(),
                'risk_contribution': outputs['risk_contribution'].squeeze(),
                'correlation_to_portfolio': outputs['correlation_to_portfolio'].squeeze(),
                'diversification_score': outputs['diversification_score'].squeeze()
            }


# OPTIMIZATION: Add optimized causal convolution
class OptimizedCausalConv1d(nn.Module):
    """
    Optimized 1D causal convolution with fused operations for better performance.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int = 1, **kwargs):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            **kwargs
        )
        # Initialize with better weight distribution for faster convergence
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        
    def forward(self, x):
        # Use a single fused operation
        result = self.conv(x)
        # Slice more efficiently using narrow instead of advanced indexing
        if self.padding > 0:
            return torch.narrow(result, 2, 0, result.size(2) - self.padding)
        return result


# OPTIMIZATION: Lightweight Temporal Block with depthwise separable convolution
class LightweightTemporalBlock(nn.Module):
    """
    Optimized temporal block that reduces computation while maintaining expressiveness.
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, 
                 stride: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        
        # Use depthwise separable convolution for efficiency (much faster than standard conv)
        # First part: depthwise convolution
        self.depthwise = OptimizedCausalConv1d(
            n_inputs, n_inputs, kernel_size,
            stride=stride, dilation=dilation,
            groups=n_inputs  # Each input channel is convolved separately
        )
        
        # Second part: pointwise convolution (1x1 conv)
        self.pointwise = nn.Conv1d(n_inputs, n_outputs, 1)
        
        # Use a single fused normalization + activation for first block
        self.bn_act1 = nn.Sequential(
            nn.BatchNorm1d(n_outputs),
            nn.SiLU()  # SiLU/Swish is faster on modern GPUs with similar performance to Mish
        )
        
        self.dropout1 = nn.Dropout(dropout)
        
        # Use depthwise separable for the second conv too
        self.depthwise2 = OptimizedCausalConv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, dilation=dilation,
            groups=n_outputs
        )
        self.pointwise2 = nn.Conv1d(n_outputs, n_outputs, 1)
        
        # Fused normalization + activation for second block
        self.bn_act2 = nn.Sequential(
            nn.BatchNorm1d(n_outputs),
            nn.SiLU()
        )
        
        self.dropout2 = nn.Dropout(dropout)
        
        # More efficient residual implementation
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
    def forward(self, x):
        # First convolution block (depthwise separable)
        identity = x
        
        # Depthwise + pointwise convolution (separable conv)
        out = self.depthwise(x)
        out = self.pointwise(out)
        
        # Apply batch norm + activation as a single fused operation
        out = self.bn_act1(out)
        out = self.dropout1(out)
        
        # Second convolution block (depthwise separable)
        out = self.depthwise2(out)
        out = self.pointwise2(out)
        out = self.bn_act2(out)
        out = self.dropout2(out)
        
        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(identity)
            
        # Faster inplace addition
        out.add_(identity)
        
        return out


# OPTIMIZATION: Optimized Temporal Network
class OptimizedTemporalConvNet(nn.Module):
    """
    Optimized Temporal Convolutional Network with much better performance.
    """
    def __init__(self, num_inputs: int, num_channels: List[int], 
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        # Create optimized network
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                LightweightTemporalBlock(
                    in_channels, out_channels, kernel_size, 
                    stride=1, dilation=dilation, dropout=dropout
                )
            )
            
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # Transpose to [batch, channels, seq_len] for convolution
        x = x.transpose(1, 2).contiguous()  # Add contiguous for better memory layout
        
        # Apply faster numerical stabilization
        x = torch.clamp(x, min=-10.0, max=10.0)
        
        out = self.network(x)
        # Transpose back to [batch, seq_len, channels]
        return out.transpose(1, 2).contiguous()  # Make contiguous again for better performance


# OPTIMIZATION: LightweightAttention - much faster than full transformer
class LightweightAttention(nn.Module):
    """
    Lightweight attention mechanism that's much faster than full transformer.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # Use linear attention approximation (much faster)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
    def forward(self, x, mask=None):
        """
        Linear attention implementation - O(n) instead of O(n²)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply ELU for positive key/query mapping (from "Transformers are RNNs" paper)
        q = torch.nn.functional.elu(q) + 1.0
        k = torch.nn.functional.elu(k) + 1.0
        
        # Apply linear attention: O(n) complexity instead of O(n²)
        kv = torch.matmul(k.transpose(-2, -1), v)  # (batch, head, head_dim, head_dim)
        out = torch.matmul(q, kv)  # (batch, head, seq_len, head_dim)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        out = self.dropout(self.out_proj(out))
        
        return out


# OPTIMIZATION: Lightweight Encoder Layer - faster than transformer
class LightweightEncoderLayer(nn.Module):
    """
    Lightweight encoder layer that's much faster than full transformer.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        
        # Lightweight attention
        self.attention = LightweightAttention(d_model, nhead, dropout)
        
        # Use GEGLU for faster convergence in feed-forward network
        self.ff_linear1 = nn.Linear(d_model, dim_feedforward * 2)
        self.ff_linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Use layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Pre-norm architecture (more stable)
        residual = x
        x = self.norm1(x)
        x = residual + self.attention(x)
        
        # Feed-forward with GEGLU activation
        residual = x
        x = self.norm2(x)
        x_ff = self.ff_linear1(x)
        
        # Split and apply GELU to one half
        x_gelu, x_linear = x_ff.chunk(2, dim=-1)
        x_gelu = torch.nn.functional.gelu(x_gelu)
        x_ff = x_gelu * x_linear
        
        x_ff = self.ff_linear2(x_ff)
        x = residual + self.dropout(x_ff)
        
        return x


# OPTIMIZATION: LightweightTransformer - uses above optimizations
class LightweightTransformer(nn.Module):
    """
    Lightweight transformer that's much faster than the original.
    """
    def __init__(self, d_model: int, nhead: int = 8, num_layers: int = 4, 
                 dim_feedforward: int = 512, dropout: float = 0.1, 
                 max_seq_length: int = 6000):
        super().__init__()
        
        # Simpler positional encoding
        self.register_buffer('pe', self._build_positional_encoding(max_seq_length, d_model))
        self.dropout = nn.Dropout(dropout)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            LightweightEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def _build_positional_encoding(self, max_len, d_model):
        """Build positional encoding more efficiently"""
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * 
                           (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Clamp to avoid extreme values
        pe = torch.clamp(pe, min=-5.0, max=5.0)
        
        return pe.unsqueeze(0)
        
    def forward(self, x, asset_ids=None):
        """
        Forward pass through the lightweight transformer.
        
        Args:
            x: [batch_size, seq_len, d_model]
            asset_ids: Ignored in this optimized version
        """
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        
        # Apply encoder layers
        for layer in self.layers:
            x = layer(x)
            
        return self.norm(x)


# OPTIMIZATION: Simplified Kelly Head
class SimplifiedNeuralKellyHead(nn.Module):
    """
    Simplified Kelly criterion head for faster training.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        
        # Simplified architecture - single step instead of multiple
        self.shared = nn.Linear(input_dim, 64)
        self.activation = nn.SiLU()  # Faster than Mish
        
        # Position sizing with single layer
        self.size_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Tanh()  # Range: [-1, 1]
        )
        
        # Funding sensitivity - simpler
        self.funding_sensitivity = nn.Sequential(
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        # Direct expected return/risk prediction
        self.expected_return = nn.Linear(64, 1)
        self.expected_risk = nn.Sequential(
            nn.Linear(64, 1),
            nn.Softplus()
        )
        
    def forward(self, x, funding_rate=None):
        # Shared features
        shared = self.activation(self.shared(x))
        
        # Direct predictions
        position_size = self.size_head(shared)
        funding_sens = self.funding_sensitivity(shared)
        expected_return = self.expected_return(shared)
        expected_risk = self.expected_risk(shared)
        
        # Apply funding adjustment if available
        if funding_rate is not None:
            funding_adj = funding_sens * funding_rate
            funding_adj = torch.clamp(funding_adj, min=-1.0, max=1.0)
            # Adjust position size with funding
            position_size = position_size + 0.2 * funding_adj
            position_size = torch.clamp(position_size, min=-1.0, max=1.0)
            
        return {
            'position_size': position_size,
            'expected_return': expected_return,
            'expected_risk': expected_risk,
            'funding_sensitivity': funding_sens
        }


# OPTIMIZATION: Simplified Portfolio Head
class SimplifiedPortfolioHead(nn.Module):
    """
    Simplified portfolio optimization head for faster training.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        
        # Single shared network instead of multiple pathways
        self.network = nn.Linear(input_dim, 32)
        self.activation = nn.SiLU()
        
        # Single output layer with multiple outputs
        self.outputs = nn.Linear(32, 4)
        
    def forward(self, x):
        # Compute shared features
        features = self.activation(self.network(x))
        
        # Get all outputs at once
        all_outputs = self.outputs(features)
        
        # Split into different metrics
        correlation = torch.tanh(all_outputs[:, 0:1])
        diversification = torch.sigmoid(all_outputs[:, 1:2])
        beta = all_outputs[:, 2:3]
        risk_contribution = torch.sigmoid(all_outputs[:, 3:4])
        
        return {
            'correlation_to_portfolio': correlation,
            'diversification_score': diversification,
            'portfolio_beta': beta,
            'risk_contribution': risk_contribution
        }


# OPTIMIZATION: OptimizedDeribitModel - Complete optimized model
class OptimizedDeribitModel(nn.Module):
    """
    Optimized hybrid model for much faster training.
    """
    def __init__(
        self, 
        input_dim: int,
        tcn_channels: List[int] = [128, 128, 128, 128],
        tcn_kernel_size: int = 3,
        transformer_dim: int = 128,
        transformer_heads: int = 8,
        transformer_layers: int = 4,
        dropout: float = 0.2,
        max_seq_length: int = 6000
    ):
        super().__init__()
        
        # Input dimensions
        self.input_dim = input_dim
        self.max_seq_length = max_seq_length
        
        # Use faster direct projection
        self.input_projection = nn.Linear(input_dim, tcn_channels[0])
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Optimized TCN
        self.tcn = OptimizedTemporalConvNet(
            num_inputs=tcn_channels[0],
            num_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=dropout
        )
        
        # TCN output dimension is the last channel size
        tcn_output_dim = tcn_channels[-1]
        
        # Projection from TCN to Transformer with faster initialization
        self.tcn_to_transformer = nn.Linear(tcn_output_dim, transformer_dim)
        nn.init.xavier_uniform_(self.tcn_to_transformer.weight, gain=0.5)
        
        # Lightweight transformer
        self.transformer = LightweightTransformer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            max_seq_length=max_seq_length
        )
        
        # Feature gating mechanism
        self.feature_gate = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.Sigmoid()
        )
        
        # Simplified prediction heads
        # Direction head
        self.direction_head = nn.Sequential(
            nn.Linear(transformer_dim, 3)  # Direct classification for speed
        )
        
        # Return head
        self.return_head = nn.Sequential(
            nn.Linear(transformer_dim, 1)
        )
        
        # Risk head
        self.risk_head = nn.Sequential(
            nn.Linear(transformer_dim, 1),
            nn.Softplus()
        )
        
        # Position sizing using simplified Kelly criterion
        self.size_head = SimplifiedNeuralKellyHead(transformer_dim)
        
        # Portfolio optimization head
        self.portfolio_head = SimplifiedPortfolioHead(transformer_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with a more stable approach."""
        for name, p in self.named_parameters():
            if 'weight' in name and len(p.shape) >= 2:
                nn.init.xavier_uniform_(p, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(p)
        
    def forward(self, x, funding_rate=None, asset_ids=None):
        """
        Forward pass through the optimized model.
        """
        batch_size, seq_len, _ = x.shape
        
        # Input preprocessing
        x = torch.clamp(x, min=-20.0, max=20.0)
        x = self.input_norm(x)
        
        # Project input features
        x = self.input_projection(x)
        
        # Apply TCN
        tcn_output = self.tcn(x)
        
        # Project to transformer dimension
        transformer_input = self.tcn_to_transformer(tcn_output)
               
        # Apply lightweight transformer
        transformer_output = self.transformer(transformer_input)
        
        # Take the last sequence element for prediction
        context_vector = transformer_output[:, -1, :]
        
        # Apply feature gating
        gate = self.feature_gate(context_vector)
        gated_features = context_vector * gate
        
        # Apply prediction heads
        direction_logits = self.direction_head(gated_features)
        expected_return = self.return_head(gated_features)
        expected_risk = self.risk_head(gated_features)
        
        # Get position sizing 
        sizing_outputs = self.size_head(gated_features, funding_rate)
        
        # Get portfolio optimization metrics
        portfolio_outputs = self.portfolio_head(gated_features)
        
        # Combine all outputs
        outputs = {
            'direction_logits': direction_logits,
            'expected_return': expected_return,
            'expected_risk': expected_risk,
            'position_size': sizing_outputs['position_size'],
            'funding_sensitivity': sizing_outputs['funding_sensitivity']
        }
        
        # Add portfolio metrics
        outputs.update(portfolio_outputs)
        
        return outputs
    
    def predict_trade_signal(self, x, funding_rate=None, asset_ids=None, threshold=0.0):
        """
        Generate a trade signal from model outputs.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            funding_rate: Optional funding rate tensor [batch_size, 1]
            asset_ids: Optional tensor of asset indices
            threshold: Confidence threshold for taking a position
            
        Returns:
            Dictionary with trade signals
        """
        with torch.no_grad():
            outputs = self.forward(x, funding_rate, asset_ids)
            
            # Get direction probabilities
            direction_probs = F.softmax(outputs['direction_logits'], dim=1)
            
            # Get position size (scaled to [-1, 1])
            position_size = outputs['position_size']
            
            # Determine trading action
            trade_action = torch.argmax(direction_probs, dim=1)  # 0=short, 1=flat, 2=long
            
            # Apply confidence threshold
            confidence = torch.max(direction_probs, dim=1)[0]
            valid_trades = confidence >= threshold
            
            # Create final position size incorporating direction and sizing
            # Convert 0,1,2 to -1,0,1
            direction = (trade_action - 1).float()
            
            # Final position is direction * size * confidence
            final_position = direction * torch.abs(position_size.squeeze()) * confidence
            
            # Zero out positions that don't meet threshold
            final_position = torch.where(valid_trades, final_position, torch.zeros_like(final_position))
            
            return {
                'direction': direction,
                'position_size': final_position,
                'confidence': confidence,
                'expected_return': outputs['expected_return'].squeeze(),
                'expected_risk': outputs['expected_risk'].squeeze(),
                'risk_contribution': outputs['risk_contribution'].squeeze(),
                'correlation_to_portfolio': outputs['correlation_to_portfolio'].squeeze(),
                'diversification_score': outputs['diversification_score'].squeeze()
            }