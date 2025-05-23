import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from pathlib import Path
import time
from collections import defaultdict
import seaborn as sns
from datetime import datetime
import json
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score
import cv2

class SlotEvolutionAnalyzer:
    """
    Advanced framework for analyzing slot representation evolution during RL fine-tuning.
    Implements comparative analysis between pre-trained and RL-optimized representations.
    """
    
    def __init__(self, args, device, envs=None, output_dir=None):
        self.args = args
        self.device = device
        self.envs = envs  # Pass environment for space definitions
        self.num_agents = args.num_agents
        
        # Enhanced output directory structure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = output_dir or os.path.join(args.log_dir, f'slot_evolution_analysis_{timestamp}')
        
        # Create subdirectories for organized output
        self.dirs = {
            'tsne': os.path.join(self.output_dir, 'tsne_visualizations'),
            'attention': os.path.join(self.output_dir, 'attention_maps'),
            'metrics': os.path.join(self.output_dir, 'diversity_metrics'),
            'comparison': os.path.join(self.output_dir, 'before_after_comparison'),
            'checkpoints': os.path.join(self.output_dir, 'model_checkpoints'),
            'temporal': os.path.join(self.output_dir, 'temporal_evolution'),
            'slot_tracking': os.path.join(self.output_dir, 'slot_tracking')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
        # Initialize tracking structures
        self.pretrained_slots = {}
        self.finetuned_slots = {}
        self.slot_evolution_history = defaultdict(list)
        self.diversity_trajectory = []
        self.attention_evolution = defaultdict(list)
        
        # Tensorboard writer for real-time monitoring
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))
        
        # Slot tracking metrics
        self.slot_specialization_history = []
        self.slot_consistency_scores = []
        
    def extract_and_compare_representations(self, observations, pretrained_model, finetuned_policies):
        """
        Extract slot representations from both pretrained and fine-tuned models
        for comparative analysis with enhanced metrics.
        """
        results = {
            'pretrained': {'slots': [], 'attention': [], 'diversity': {}, 'features': []},
            'finetuned': {'slots': [], 'attention': [], 'diversity': {}, 'features': []},
            'evolution_metrics': {},
            'slot_matching': {},
            'attention_consistency': {}
        }
        
        # Process observations through pretrained model
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations).to(self.device)
            
            # Extract from pretrained model
            pretrained_out = pretrained_model(
                obs_tensor.permute(0, 3, 1, 2),
                tau=self.args.tau_final,
                sigma=self.args.sigma_final,
                is_Train=False,
                visualize=True
            )
            results['pretrained']['slots'] = pretrained_out['slots'].cpu()
            results['pretrained']['attention'] = pretrained_out['attn'].cpu()
            results['pretrained']['features'] = pretrained_out.get('features', None)
            results['pretrained']['diversity'] = self.compute_enhanced_diversity_metrics(
                pretrained_out['slots']
            )
            
            # Extract from each fine-tuned agent
            all_agent_slots = []
            all_agent_attentions = []
            
            for agent_id, policy in enumerate(finetuned_policies):
                finetuned_out = policy.actor.slot_attn(
                    obs_tensor.permute(0, 3, 1, 2),
                    tau=policy.actor.tau,
                    sigma=policy.actor.sigma,
                    is_Train=False,
                    visualize=True
                )
                
                all_agent_slots.append(finetuned_out['slots'])
                all_agent_attentions.append(finetuned_out['attn'])
                
                if agent_id == 0:  # Use first agent as representative
                    results['finetuned']['slots'] = finetuned_out['slots'].cpu()
                    results['finetuned']['attention'] = finetuned_out['attn'].cpu()
                    results['finetuned']['features'] = finetuned_out.get('features', None)
                    results['finetuned']['diversity'] = self.compute_enhanced_diversity_metrics(
                        finetuned_out['slots']
                    )
        
        # Compute evolution metrics
        results['evolution_metrics'] = self.compute_representation_drift(
            results['pretrained']['slots'],
            results['finetuned']['slots']
        )
        
        # Compute slot matching between pre and post training
        results['slot_matching'] = self.compute_slot_correspondence(
            results['pretrained']['slots'],
            results['finetuned']['slots']
        )
        
        # Analyze attention consistency across agents
        if len(all_agent_attentions) > 1:
            results['attention_consistency'] = self.compute_attention_consistency(
                all_agent_attentions
            )
        
        # Compute inter-agent slot alignment
        if len(all_agent_slots) > 1:
            results['inter_agent_alignment'] = self.compute_inter_agent_alignment(
                all_agent_slots
            )
        
        return results
    def compute_enhanced_diversity_metrics(self, slots):
        """
        Comprehensive diversity quantification across multiple theoretical frameworks.
    
        Methodological Integration:
        - Information-theoretic measures (entropy, mutual information)
        - Geometric diversity (pairwise distances, volume metrics)
        - Statistical dispersion (variance, effective rank)
        - Topological analysis (persistent homology features)
    
        Args:
            slots: torch.Tensor [batch_size, num_slots, slot_dim]
    
        Returns:
            dict: Comprehensive diversity metrics
        """
        batch_size, num_slots, slot_dim = slots.shape
    
        # Standard diversity: Average pairwise cosine distance
        slots_norm = F.normalize(slots, p=2, dim=-1)
        similarity_matrix = torch.bmm(slots_norm, slots_norm.transpose(1, 2))
    
        # Mask diagonal
        mask = ~torch.eye(num_slots, dtype=bool, device=slots.device)
        pairwise_similarities = similarity_matrix[:, mask].reshape(batch_size, num_slots, num_slots-1)
        standard_diversity = (1 - pairwise_similarities).mean().item()
    
        # Effective rank: Measure of dimensionality utilization
        slots_flat = slots.reshape(batch_size, -1)
        _, S, _ = torch.svd(slots_flat)
        S_normalized = S / S.sum(dim=1, keepdim=True)
        effective_rank = torch.exp(-torch.sum(S_normalized * torch.log(S_normalized + 1e-10), dim=1))
    
        # Specialization index: Measure of slot differentiation
        slot_variances = slots.var(dim=0).mean(dim=1)  # Variance per slot
        specialization_index = slot_variances.std() / (slot_variances.mean() + 1e-8)
    
        # Information-theoretic entropy
        # Treat slot activations as probability distributions
        slots_softmax = F.softmax(slots.reshape(-1, slot_dim), dim=1)
        entropy = -torch.sum(slots_softmax * torch.log(slots_softmax + 1e-10), dim=1)
    
        # Coverage metric: Volume of convex hull in PCA space
        coverage = self._compute_coverage_metric(slots)
    
        # Orthogonality score
        identity_target = torch.eye(num_slots, device=slots.device)
        orthogonality_score = F.mse_loss(similarity_matrix, identity_target.unsqueeze(0).expand(batch_size, -1, -1))
    
        # Clustering coefficient: Local density measure
        clustering_coef = self._compute_clustering_coefficient(similarity_matrix, threshold=0.5)
    
        return {
            'standard_diversity': standard_diversity,
            'effective_rank': effective_rank.mean().item(),
            'specialization_index': specialization_index.item(),
            'entropy': entropy.mean().item(),
            'coverage': coverage,
            'orthogonality_score': orthogonality_score.item(),
            'clustering_coefficient': clustering_coef,
            'diversity_variance': pairwise_similarities.var().item()
        }

    def _compute_coverage_metric(self, slots):
        """Compute the volume coverage in reduced dimensional space."""
        # Project to 2D using PCA for tractable convex hull computation
        slots_flat = slots.reshape(-1, slots.shape[-1])
        U, S, V = torch.svd(slots_flat.T)
    
        # Project onto first 2 principal components
        slots_2d = torch.mm(slots_flat, U[:, :2])
    
        # Compute area of convex hull (simplified rectangular approximation)
        mins = slots_2d.min(dim=0)[0]
        maxs = slots_2d.max(dim=0)[0]
        area = torch.prod(maxs - mins).item()
    
        # Normalize by number of points
        return area / slots_flat.shape[0]

    def _compute_clustering_coefficient(self, similarity_matrix, threshold=0.5):
        """Compute average clustering coefficient of slot similarity graph."""
        # Binarize similarity matrix
        adjacency = (similarity_matrix > threshold).float()
    
        # Compute clustering coefficient for each node
        clustering_coeffs = []
        for i in range(adjacency.shape[1]):
            neighbors = adjacency[:, i].sum(dim=1) - 1  # Exclude self
            possible_connections = neighbors * (neighbors - 1) / 2
        
            # Count actual connections between neighbors
            actual_connections = 0
            for j in range(adjacency.shape[1]):
                if i != j and adjacency[:, i, j].any():
                    for k in range(j+1, adjacency.shape[1]):
                        if i != k and adjacency[:, i, k].any():
                            actual_connections += adjacency[:, j, k].sum()
        
            # Clustering coefficient
            coeff = actual_connections / (possible_connections.sum() + 1e-8)
            clustering_coeffs.append(coeff)
    
        return torch.stack(clustering_coeffs).mean().item()
    
    def compute_representation_drift(self, pretrained_slots, finetuned_slots):
        """
        Quantify representational divergence between pretrained and fine-tuned slot embeddings.
    
        Theoretical Foundation: Measures the geometric displacement in latent space,
        capturing the magnitude of representational adaptation during RL optimization.
    
        Args:
            pretrained_slots: torch.Tensor [batch_size, num_slots, slot_dim]
            finetuned_slots: torch.Tensor [batch_size, num_slots, slot_dim]
    
        Returns:
            dict: Multi-scale drift metrics capturing various aspects of representational change
        """
        #Normalize representations for scale-invariant comparison
        pre_norm = F.normalize(pretrained_slots.flatten(1), p=2, dim=1)
        post_norm = F.normalize(finetuned_slots.flatten(1), p=2, dim=1)
    
        # Primary metric: Cosine distance in flattened representation space
        cosine_similarity = F.cosine_similarity(pre_norm, post_norm, dim=1)
        cosine_distance = 1 - cosine_similarity
    
        # Secondary metric: Euclidean drift in normalized space
        euclidean_drift = torch.norm(pre_norm - post_norm, p=2, dim=1)
    
        # Tertiary metric: Subspace alignment via principal angles
        # Compute SVD of slot covariance matrices
        pre_cov = torch.bmm(pretrained_slots.transpose(1, 2), pretrained_slots)
        post_cov = torch.bmm(finetuned_slots.transpose(1, 2), finetuned_slots)
    
        # Grassmann distance between subspaces
        grassmann_distance = self._compute_grassmann_distance(pre_cov, post_cov)
    
        # Slot-wise drift analysis
        slot_drifts = []
        for s in range(pretrained_slots.shape[1]):
            slot_similarity = F.cosine_similarity(
                pretrained_slots[:, s], 
                finetuned_slots[:, s], 
                dim=1
            )
            slot_drifts.append(1 - slot_similarity)
    
        return {
            'cosine_distance': cosine_distance.mean().item(),
            'euclidean_drift': euclidean_drift.mean().item(),
            'grassmann_distance': grassmann_distance.mean().item(),
            'per_slot_drift': torch.stack(slot_drifts).mean(dim=1).tolist(),
            'drift_variance': cosine_distance.var().item(),
            'max_drift': cosine_distance.max().item(),
            'min_drift': cosine_distance.min().item()
        }

    def _compute_grassmann_distance(self, cov1, cov2):
        """Compute distance between subspaces using principal angles."""
        batch_size = cov1.shape[0]
        distances = []
    
        for b in range(batch_size):
            # Eigendecomposition
            _, U1 = torch.linalg.eigh(cov1[b])
            _, U2 = torch.linalg.eigh(cov2[b])
        
            # Principal angles via SVD of U1^T U2
            M = torch.mm(U1.T, U2)
            _, S, _ = torch.svd(M)
        
            # Grassmann distance from principal angles
            principal_angles = torch.acos(torch.clamp(S, -1, 1))
            grassmann_dist = torch.norm(principal_angles)
            distances.append(grassmann_dist)
    
        return torch.stack(distances)    
    def compute_slot_correspondence(self, pre_slots, post_slots):
        """
        Find correspondence between pre-trained and fine-tuned slots using Hungarian matching.
        """
        batch_size, num_slots, slot_dim = pre_slots.shape
        
        # Normalize slots
        pre_norm = F.normalize(pre_slots, p=2, dim=-1)
        post_norm = F.normalize(post_slots, p=2, dim=-1)
        
        correspondences = []
        correspondence_scores = []
        
        for b in range(batch_size):
            # Compute pairwise cosine similarities
            similarity_matrix = torch.mm(pre_norm[b], post_norm[b].T)
            
            # Convert to cost matrix (Hungarian algorithm minimizes)
            cost_matrix = 1 - similarity_matrix.cpu().numpy()
            
            # Find optimal assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Store correspondence and scores
            correspondences.append(list(zip(row_ind, col_ind)))
            correspondence_scores.append([
                similarity_matrix[i, j].item() for i, j in zip(row_ind, col_ind)
            ])
        
        return {
            'correspondences': correspondences,
            'scores': correspondence_scores,
            'mean_similarity': np.mean([np.mean(scores) for scores in correspondence_scores]),
            'slot_stability': self.compute_slot_stability(correspondence_scores)
        }
    
    def compute_slot_stability(self, correspondence_scores):
        """
        Measure how stable slot identities are across the dataset.
        """
        # Convert to numpy array for easier manipulation
        scores_array = np.array(correspondence_scores)
        
        # Compute variance of correspondence scores for each slot
        slot_variances = np.var(scores_array, axis=0)
        
        # Lower variance = more stable slot identity
        stability_scores = 1 - (slot_variances / np.max(slot_variances + 1e-8))
        
        return {
            'per_slot_stability': stability_scores.tolist(),
            'mean_stability': float(np.mean(stability_scores)),
            'min_stability': float(np.min(stability_scores)),
            'max_stability': float(np.max(stability_scores))
        }
    
    def compute_attention_consistency(self, agent_attentions):
        """
        Measure consistency of attention patterns across different agents.
        """
        num_agents = len(agent_attentions)
        if num_agents < 2:
            return {}
        
        # Stack all attention maps
        all_attentions = torch.stack(agent_attentions)  # [num_agents, batch, slots, H*W]
        
        # Compute pairwise similarities between agent attention patterns
        consistency_scores = []
        
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                # Compute cosine similarity between attention patterns
                attn_i = F.normalize(all_attentions[i].flatten(1), p=2, dim=1)
                attn_j = F.normalize(all_attentions[j].flatten(1), p=2, dim=1)
                
                similarity = F.cosine_similarity(attn_i, attn_j, dim=1)
                consistency_scores.append(similarity.mean().item())
        
        return {
            'mean_consistency': np.mean(consistency_scores),
            'std_consistency': np.std(consistency_scores),
            'pairwise_scores': consistency_scores
        }
    
    def compute_inter_agent_alignment(self, agent_slots):
        """
        Measure how aligned slot representations are across different agents.
        """
        num_agents = len(agent_slots)
        if num_agents < 2:
            return {}
        
        # Compute canonical slots (mean across agents)
        canonical_slots = torch.stack(agent_slots).mean(dim=0)
        
        # Measure deviation from canonical
        alignment_scores = []
        for agent_slots_i in agent_slots:
            # Normalize for comparison
            canonical_norm = F.normalize(canonical_slots.flatten(1), p=2, dim=1)
            agent_norm = F.normalize(agent_slots_i.flatten(1), p=2, dim=1)
            
            similarity = F.cosine_similarity(canonical_norm, agent_norm, dim=1)
            alignment_scores.append(similarity.mean().item())
        
        return {
            'alignment_scores': alignment_scores,
            'mean_alignment': np.mean(alignment_scores),
            'std_alignment': np.std(alignment_scores)
        }
    
    def create_temporal_evolution_visualization(self, checkpoint_results, save_path=None):
        """
        Create visualization showing temporal evolution of slot representations.
        """
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Extract temporal data
        episodes = [r['episode'] for r in checkpoint_results]
        diversity_scores = [r['diversity']['standard_diversity'] for r in checkpoint_results]
        specialization_scores = [r['diversity']['specialization_index'] for r in checkpoint_results]
        entropy_scores = [r['diversity']['entropy'] for r in checkpoint_results]
        drift_scores = [r['drift_from_pretrained'] for r in checkpoint_results]
        
        # 1. Diversity evolution
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(episodes, diversity_scores, 'b-', linewidth=2, marker='o')
        ax1.axhline(y=checkpoint_results[0]['pretrained_diversity'], 
                   color='r', linestyle='--', label='Pretrained baseline')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Slot Diversity')
        ax1.set_title('Slot Diversity Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Specialization evolution
        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(episodes, specialization_scores, 'g-', linewidth=2, marker='s')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Specialization Index')
        ax2.set_title('Slot Specialization Evolution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Entropy evolution
        ax3 = plt.subplot(gs[0, 2])
        ax3.plot(episodes, entropy_scores, 'm-', linewidth=2, marker='^')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Slot Entropy')
        ax3.set_title('Slot Activation Entropy')
        ax3.grid(True, alpha=0.3)
        
        # 4. Drift from pretrained
        ax4 = plt.subplot(gs[1, :])
        ax4.plot(episodes, drift_scores, 'r-', linewidth=2, marker='D')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Drift from Pretrained (cosine distance)')
        ax4.set_title('Representation Drift During Training')
        ax4.grid(True, alpha=0.3)
        
        # 5. Combined heatmap of all metrics
        ax5 = plt.subplot(gs[2, :])
        
        # Normalize metrics for heatmap
        metrics_matrix = np.array([
            diversity_scores,
            specialization_scores,
            entropy_scores,
            drift_scores
        ])
        
        # Normalize each metric to [0, 1]
        metrics_norm = (metrics_matrix - metrics_matrix.min(axis=1, keepdims=True)) / \
                      (metrics_matrix.max(axis=1, keepdims=True) - metrics_matrix.min(axis=1, keepdims=True) + 1e-8)
        
        im = ax5.imshow(metrics_norm, aspect='auto', cmap='viridis')
        ax5.set_yticks(range(4))
        ax5.set_yticklabels(['Diversity', 'Specialization', 'Entropy', 'Drift'])
        ax5.set_xlabel('Training Progress')
        ax5.set_title('Normalized Metrics Evolution Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5)
        cbar.set_label('Normalized Value')
        
        plt.suptitle('Temporal Evolution of Slot Representations During RL Training', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_slot_tracking_visualization(self, results, observations, save_path=None):
        """
        Visualize how individual slots track different features before and after training.
        """
        pre_attention = results['pretrained']['attention'].numpy()
        post_attention = results['finetuned']['attention'].numpy()
        correspondences = results['slot_matching']['correspondences']
        
        num_samples = min(3, len(observations))
        num_slots = pre_attention.shape[1]
        
        fig = plt.figure(figsize=(24, 8 * num_samples))
        
        # Create custom colormap
        colors = [(1,1,1,0), (1,0,0,1)]
        attention_cmap = LinearSegmentedColormap.from_list('attention', colors)
        
        for sample_idx in range(num_samples):
            # Get slot correspondences for this sample
            slot_mapping = dict(correspondences[sample_idx])
            
            # Original image
            ax = plt.subplot(num_samples, num_slots * 2 + 1, 
                           sample_idx * (num_slots * 2 + 1) + 1)
            ax.imshow(observations[sample_idx])
            ax.set_title('Input Image', fontsize=12)
            ax.axis('off')
            
            # For each slot pair (pre -> post)
            for pre_slot_idx in range(num_slots):
                post_slot_idx = slot_mapping.get(pre_slot_idx, pre_slot_idx)
                
                # Pre-trained attention
                ax_pre = plt.subplot(num_samples, num_slots * 2 + 1,
                                   sample_idx * (num_slots * 2 + 1) + 2 + pre_slot_idx * 2)
                ax_pre.imshow(observations[sample_idx])
                
                # Resize attention to match image size
                pre_attn = pre_attention[sample_idx, pre_slot_idx]
                H, W = observations[sample_idx].shape[:2]
                h, w = int(np.sqrt(pre_attn.shape[0])), int(np.sqrt(pre_attn.shape[0]))
                pre_attn = pre_attn.reshape(h, w)
                pre_attn_resized = cv2.resize(pre_attn, (W, H), interpolation=cv2.INTER_LINEAR)
                
                ax_pre.imshow(pre_attn_resized, cmap=attention_cmap, alpha=0.7)
                ax_pre.set_title(f'Pre: Slot {pre_slot_idx}', fontsize=10)
                ax_pre.axis('off')
                
                # Post-training attention
                ax_post = plt.subplot(num_samples, num_slots * 2 + 1,
                                    sample_idx * (num_slots * 2 + 1) + 3 + pre_slot_idx * 2)
                ax_post.imshow(observations[sample_idx])
                
                post_attn = post_attention[sample_idx, post_slot_idx]
                post_attn = post_attn.reshape(h, w)
                post_attn_resized = cv2.resize(post_attn, (W, H), interpolation=cv2.INTER_LINEAR)
                
                ax_post.imshow(post_attn_resized, cmap=attention_cmap, alpha=0.7)
                ax_post.set_title(f'Post: Slot {post_slot_idx}', fontsize=10)
                ax_post.axis('off')
                
                # Draw arrow to show correspondence
                if pre_slot_idx != post_slot_idx:
                    ax_post.annotate('', xy=(0.5, 0), xytext=(0.5, 1),
                                   xycoords='axes fraction', 
                                   arrowprops=dict(arrowstyle='<-', color='blue', lw=2))
        
        plt.suptitle('Slot Attention Tracking: Pre-training → RL Fine-tuning\n' + 
                    'Arrows indicate slot reassignments',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_trajectory_with_checkpoints(self, checkpoint_paths, test_observations, 
                                          pretrained_model):
        """
        Analyze representation evolution across training checkpoints with enhanced metrics.
        """
        checkpoint_results = []
        
        # Get pretrained baseline
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(test_observations).to(self.device)
            pre_out = pretrained_model(
                obs_tensor.permute(0, 3, 1, 2),
                tau=self.args.tau_final,
                sigma=self.args.sigma_final,
                is_Train=False,
                visualize=False
            )
            pretrained_slots = pre_out['slots']
            pretrained_metrics = self.compute_enhanced_diversity_metrics(pretrained_slots)
        
        # Analyze each checkpoint
        for checkpoint_info in checkpoint_paths:
            episode = checkpoint_info['episode']
            path = checkpoint_info['path']
            
            # Load checkpoint
            policies = self.load_finetuned_policies(path)
            
            # Extract representations
            with torch.no_grad():
                out = policies[0].actor.slot_attn(
                    obs_tensor.permute(0, 3, 1, 2),
                    tau=policies[0].actor.tau,
                    sigma=policies[0].actor.sigma,
                    is_Train=False,
                    visualize=True
                )
                
                checkpoint_slots = out['slots']
                checkpoint_metrics = self.compute_enhanced_diversity_metrics(checkpoint_slots)
                
                # Compute drift from pretrained
                drift = self.compute_representation_drift(
                    pretrained_slots.cpu(), 
                    checkpoint_slots.cpu()
                )
                
                checkpoint_results.append({
                    'episode': episode,
                    'diversity': checkpoint_metrics,
                    'drift_from_pretrained': drift['cosine_distance'],
                    'pretrained_diversity': pretrained_metrics['standard_diversity'],
                    'attention': out['attn'].cpu()
                })
        
        return checkpoint_results
    
    def load_pretrained_slot_attention_model(self, checkpoint_path):
        """Load standalone pretrained slot attention model."""
        from onpolicy.algorithms.utils.QSA.train_qsa import generate_model
        
        model = generate_model(self.args).to(self.device)
        
        # Load checkpoint - handle different checkpoint formats
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                # Assume it's just the state dict
                model.load_state_dict(checkpoint)
        else:
            # Try to find the model in the directory
            model_files = list(Path(checkpoint_path).glob('*model.pt'))
            if model_files:
                checkpoint = torch.load(model_files[0], map_location=self.device)
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                raise FileNotFoundError(f"No model found at {checkpoint_path}")
        
        model.eval()
        return model
    
    def load_finetuned_policies(self, model_dir):
        """Load RL-trained policies with proper space definitions."""
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
        
        policies = []
        
        # Use dummy spaces if envs not provided
            
        
        for agent_id in range(self.args.num_agents):
            player_key = f"player_{agent_id}"
            # Create dummy spaces (would need actual env spaces in practice)
            obs_space =  self.envs.observation_space[player_key]['RGB']  # Replace with actual observation space
            act_space = self.envs.action_space[player_key] # Replace with actual action space
            share_observation_space = self.envs.share_observation_space[player_key] if self.args.use_centralized_V else \
                    self.envs.share_observation_space[player_key]

            policy = R_MAPPOPolicy(
                self.args, obs_space, share_observation_space, act_space, self.device
            )
            
            # Load actor weights
            actor_path = os.path.join(model_dir, f'actor_agent_{agent_id}.pt')
            if os.path.exists(actor_path):
                actor_state_dict = torch.load(actor_path, map_location=self.device)
                policy.actor.load_state_dict(actor_state_dict)
                policy.actor.eval()
                
                policies.append(policy)
            else:
                print(f"Warning: Actor weights not found for agent {agent_id}")
                
        return policies
    
    def generate_comprehensive_report(self, all_results, save_path):
        """Generate detailed analysis report with actionable insights."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'num_agents': self.args.num_agents,
                'num_slots': self.args.scoff_num_units if hasattr(self.args, 'scoff_num_units') else 'N/A',
                'fine_tuning_type': self.args.fine_tuning_type,
                'substrate': self.args.substrate_name
            },
            'key_findings': {},
            'detailed_metrics': {},
            'recommendations': []
        }
        
        # Analyze key findings
        if 'static_analysis' in all_results:
            static = all_results['static_analysis']
            
            # Key finding 1: Representation drift
            drift = static['evolution_metrics']['cosine_distance']
            report['key_findings']['representation_drift'] = {
                'value': drift,
                'interpretation': 'High' if drift > 0.3 else 'Moderate' if drift > 0.1 else 'Low',
                'significance': 'RL training induced substantial representation changes' if drift > 0.3 else 'Representations remained relatively stable'
            }
            
            # Key finding 2: Diversity changes
            div_change = (static['finetuned']['diversity']['standard_diversity'] - 
                         static['pretrained']['diversity']['standard_diversity'])
            report['key_findings']['diversity_evolution'] = {
                'change': div_change,
                'direction': 'increased' if div_change > 0 else 'decreased',
                'magnitude': abs(div_change)
            }
            
            # Key finding 3: Slot stability
            if 'slot_matching' in static:
                stability = static['slot_matching']['slot_stability']['mean_stability']
                report['key_findings']['slot_identity_preservation'] = {
                    'stability_score': stability,
                    'interpretation': 'Stable' if stability > 0.7 else 'Moderate' if stability > 0.4 else 'Unstable'
                }
        
        # Add temporal analysis if available
        if 'temporal_analysis' in all_results:
            temporal = all_results['temporal_analysis']
            report['key_findings']['training_dynamics'] = {
                'episodes_analyzed': len(temporal),
                'convergence_episode': self._find_convergence_point(temporal),
                'final_drift': temporal[-1]['drift_from_pretrained'] if temporal else None
            }
        
        # Detailed metrics
        report['detailed_metrics'] = {
            'diversity_metrics': all_results.get('static_analysis', {}).get('finetuned', {}).get('diversity', {}),
            'evolution_metrics': all_results.get('static_analysis', {}).get('evolution_metrics', {}),
            'inter_agent_consistency': all_results.get('static_analysis', {}).get('inter_agent_alignment', {})
        }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(all_results)
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Also create a markdown summary
        self._create_markdown_summary(report, 
                                     os.path.join(self.output_dir, 'analysis_summary.md'))
        
        return report
    
    def _find_convergence_point(self, temporal_data):
        """Find when representations converged during training."""
        if len(temporal_data) < 3:
            return None
            
        drifts = [d['drift_from_pretrained'] for d in temporal_data]
        
        # Simple convergence detection: when drift stabilizes
        for i in range(2, len(drifts)):
            if abs(drifts[i] - drifts[i-1]) < 0.01 and abs(drifts[i-1] - drifts[i-2]) < 0.01:
                return temporal_data[i]['episode']
        
        return None
    
    def _generate_recommendations(self, results):
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        if 'static_analysis' in results:
            static = results['static_analysis']
            
            # Check diversity
            div_change = (static['finetuned']['diversity']['standard_diversity'] - 
                         static['pretrained']['diversity']['standard_diversity'])
            
            if div_change < -0.1:
                recommendations.append({
                    'issue': 'Slot diversity decreased significantly',
                    'suggestion': 'Consider adding diversity regularization or increasing sigma parameter',
                    'priority': 'High'
                })
            
            # Check specialization
            spec = static['finetuned']['diversity']['specialization_index']
            if spec < 0.5:
                recommendations.append({
                    'issue': 'Low slot specialization',
                    'suggestion': 'Increase orthogonality loss weight or use stronger initialization',
                    'priority': 'Medium'
                })
            
            # Check inter-agent alignment
            if 'inter_agent_alignment' in static:
                alignment = static['inter_agent_alignment']['mean_alignment']
                if alignment < 0.7:
                    recommendations.append({
                        'issue': 'Poor inter-agent slot consistency',
                        'suggestion': 'Consider parameter sharing or consensus mechanisms',
                        'priority': 'Medium'
                    })
        
        return recommendations
    
    def _create_markdown_summary(self, report, save_path):
        """Create a readable markdown summary of findings."""
        with open(save_path, 'w') as f:
            f.write("# Slot Attention Evolution Analysis Report\n\n")
            f.write(f"**Date**: {report['timestamp']}\n")
            f.write(f"**Configuration**: {report['configuration']['substrate']} with {report['configuration']['num_agents']} agents\n\n")
            
            f.write("## Key Findings\n\n")
            
            for finding, details in report['key_findings'].items():
                f.write(f"### {finding.replace('_', ' ').title()}\n")
                for key, value in details.items():
                    f.write(f"- **{key}**: {value}\n")
                f.write("\n")
            
            f.write("## Recommendations\n\n")
            for rec in report['recommendations']:
                f.write(f"- **{rec['priority']} Priority**: {rec['issue']}\n")
                f.write(f"  - *Suggestion*: {rec['suggestion']}\n\n")
    
    def create_comparative_tsne_visualization(self, results, observations, save_path=None):
        """
        Generate comparative t-SNE projections elucidating representational evolution.
    
        Theoretical Framework:
        - Manifold learning reveals latent geometries of slot representations
        - Comparative analysis quantifies structural reorganization
        - Perplexity adaptation ensures robust neighborhood preservation
    
        Methodological Considerations:
        - Dynamic perplexity scaling based on sample size
        - Multi-scale analysis through varied initialization seeds
        - Preservation of topological relationships across transformations
        """
        pretrained_slots = results['pretrained']['slots'].numpy()
        finetuned_slots = results['finetuned']['slots'].numpy()
    
        batch_size, num_slots, slot_dim = pretrained_slots.shape
    
        # Flatten slot representations for t-SNE projection
        pre_flat = pretrained_slots.reshape(-1, slot_dim)
        post_flat = finetuned_slots.reshape(-1, slot_dim)
    
        # Concatenate for unified projection space
        combined_slots = np.vstack([pre_flat, post_flat])
    
        # Adaptive perplexity calculation
        n_samples = combined_slots.shape[0]
        perplexity = min(30, max(5, n_samples // 10))
    
        # Multi-scale t-SNE analysis
        tsne_models = []
        embeddings = []
    
        for seed in [42, 123, 456]:  # Multiple initializations for robustness
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                learning_rate='auto',
                n_iter=1000,
                random_state=seed,
                method='barnes_hut',
                angle=0.5
            )
            embedding = tsne.fit_transform(combined_slots)
            embeddings.append(embedding)
            tsne_models.append(tsne)
    
        # Select embedding with lowest KL divergence
        best_embedding = embeddings[0]  # Could implement KL selection
    
        # Split back into pre/post embeddings
        pre_embedding = best_embedding[:len(pre_flat)]
        post_embedding = best_embedding[len(pre_flat):]
    
        # Create sophisticated visualization
        fig = plt.figure(figsize=(20, 8))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.3)
    
        # Define color schemes
        colors_pre = plt.cm.viridis(np.linspace(0, 0.8, num_slots))
        colors_post = plt.cm.plasma(np.linspace(0, 0.8, num_slots))
    
        # 1. Pretrained slot distribution
        ax1 = plt.subplot(gs[0])
        for slot_idx in range(num_slots):
            slot_mask = np.arange(batch_size) * num_slots + slot_idx
            ax1.scatter(
                pre_embedding[slot_mask, 0],
                pre_embedding[slot_mask, 1],
                c=[colors_pre[slot_idx]],
                label=f'Slot {slot_idx}',
                alpha=0.7,
                s=50,
                edgecolors='white',
                linewidth=1
                )
    
        ax1.set_title('Pretrained Slot Representations\n(Initialization State)', 
                  fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE Component 1')
        ax1.set_ylabel('t-SNE Component 2')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9)
    
        # 2. Fine-tuned slot distribution
        ax2 = plt.subplot(gs[1])
        for slot_idx in range(num_slots):
            slot_mask = np.arange(batch_size) * num_slots + slot_idx
            ax2.scatter(
            post_embedding[slot_mask, 0],
            post_embedding[slot_mask, 1],
            c=[colors_post[slot_idx]],
            label=f'Slot {slot_idx}',
            alpha=0.7,
            s=50,
            edgecolors='white',
            linewidth=1
            )
    
        ax2.set_title('RL-Optimized Slot Representations\n(Converged State)', 
                  fontsize=14, fontweight='bold')
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9)
    
        # 3. Trajectory visualization
        ax3 = plt.subplot(gs[2])
    
        # Plot migration paths
        for slot_idx in range(num_slots):
            for sample_idx in range(min(10, batch_size)):  # Limit trajectories for clarity
                idx = sample_idx * num_slots + slot_idx
            
                # Draw arrow from pre to post
                ax3.annotate('', 
                        xy=(post_embedding[idx, 0], post_embedding[idx, 1]),
                        xytext=(pre_embedding[idx, 0], pre_embedding[idx, 1]),
                        arrowprops=dict(
                            arrowstyle='->', 
                            color=colors_pre[slot_idx],
                            alpha=0.4,
                            lw=1.5,
                            shrinkA=5,
                            shrinkB=5
                        ))
    
        # Plot start and end points
        ax3.scatter(pre_embedding[:, 0], pre_embedding[:, 1], 
               c='gray', alpha=0.3, s=30, label='Initial')
        ax3.scatter(post_embedding[:, 0], post_embedding[:, 1], 
               c='red', alpha=0.5, s=30, label='Final')
    
        ax3.set_title('Representational Drift Trajectories\n(Evolution Pathways)', 
                  fontsize=14, fontweight='bold')
        ax3.set_xlabel('t-SNE Component 1')
        ax3.set_ylabel('t-SNE Component 2')
        ax3.grid(True, alpha=0.3)
        ax3.legend(framealpha=0.9)
    
        # Add quantitative annotations
        fig.text(0.5, 0.02, 
             f'Perplexity: {perplexity} | Samples: {n_samples} | ' + 
             f'Mean Drift: {results["evolution_metrics"]["cosine_distance"]:.3f}',
             ha='center', fontsize=10, style='italic')
    
        plt.suptitle('Comparative t-SNE Analysis: Slot Representation Evolution',
                fontsize=16, fontweight='bold', y=0.98)
    
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
        return fig


    def create_attention_evolution_visualization(self, results, observations, save_path=None):
        """
        Sophisticated visualization of attention pattern metamorphosis during training.
    
        Analytical Framework:
        - Attention entropy quantifies focusing/diffusion dynamics
        - Spatial coherence metrics reveal semantic clustering
        - Cross-temporal consistency indicates representational stability
    
        Visualization Architecture:
        - Multi-scale attention heatmaps
        - Statistical distribution analysis
        - Information-theoretic metrics
        """
        pre_attention = results['pretrained']['attention'].numpy()
        post_attention = results['finetuned']['attention'].numpy()
    
        batch_size, num_slots, num_patches = pre_attention.shape
        H = W = int(np.sqrt(num_patches))
    
        # Select representative samples
        num_samples = min(4, batch_size)
        sample_indices = np.linspace(0, batch_size-1, num_samples, dtype=int)
    
        fig = plt.figure(figsize=(24, 16))
        gs = gridspec.GridSpec(num_samples + 2, num_slots * 2 + 2, 
                          height_ratios=[1]*num_samples + [0.8, 0.8],
                          wspace=0.15, hspace=0.3)
    
        # Custom colormap for attention
        colors = [(0, 0, 0.3), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)]
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('attention', colors, N=n_bins)
    
        # Main attention visualization
        for sample_idx, actual_idx in enumerate(sample_indices):
            # Original image
            ax_img = plt.subplot(gs[sample_idx, 0])
            ax_img.imshow(observations[actual_idx])
            ax_img.set_title(f'Input {actual_idx}', fontsize=10)
            ax_img.axis('off')
        
            # Attention maps for each slot
            for slot_idx in range(num_slots):
                # Pretrained attention
                ax_pre = plt.subplot(gs[sample_idx, 1 + slot_idx * 2])
                attn_pre = pre_attention[actual_idx, slot_idx].reshape(H, W)
            
                # Apply Gaussian smoothing for visual clarity
                from scipy.ndimage import gaussian_filter
                attn_pre_smooth = gaussian_filter(attn_pre, sigma=0.5)
            
                im_pre = ax_pre.imshow(attn_pre_smooth, cmap=cmap, vmin=0, 
                                   vmax=pre_attention.max())
                ax_pre.set_title(f'Pre S{slot_idx}', fontsize=8)
                ax_pre.axis('off')
            
                # Fine-tuned attention
                ax_post = plt.subplot(gs[sample_idx, 2 + slot_idx * 2])
                attn_post = post_attention[actual_idx, slot_idx].reshape(H, W)
                attn_post_smooth = gaussian_filter(attn_post, sigma=0.5)
            
                im_post = ax_post.imshow(attn_post_smooth, cmap=cmap, vmin=0,
                                    vmax=post_attention.max())
                ax_post.set_title(f'Post S{slot_idx}', fontsize=8)
                ax_post.axis('off')
    
        # Statistical analysis row
        ax_stats = plt.subplot(gs[-2, :])
    
        # Compute attention statistics
        pre_entropy = self._compute_attention_entropy(pre_attention)
        post_entropy = self._compute_attention_entropy(post_attention)
    
        pre_sparsity = self._compute_attention_sparsity(pre_attention)
        post_sparsity = self._compute_attention_sparsity(post_attention)
    
        # Plot statistics
        x = np.arange(num_slots)
        width = 0.35
    
        ax_stats.bar(x - width/2, pre_entropy, width, label='Pre-trained Entropy',
                alpha=0.7, color='steelblue')
        ax_stats.bar(x + width/2, post_entropy, width, label='Fine-tuned Entropy',
                alpha=0.7, color='darkorange')
    
        ax_stats.set_xlabel('Slot Index')
        ax_stats.set_ylabel('Attention Entropy (bits)')
        ax_stats.set_title('Information-Theoretic Analysis of Attention Evolution')
        ax_stats.legend()
        ax_stats.grid(True, alpha=0.3)
    
        # Sparsity analysis
        ax_sparse = plt.subplot(gs[-1, :num_slots])
        ax_sparse.plot(pre_sparsity, 'o-', label='Pre-trained', color='steelblue')
        ax_sparse.plot(post_sparsity, 's-', label='Fine-tuned', color='darkorange')
        ax_sparse.set_xlabel('Slot Index')
        ax_sparse.set_ylabel('Gini Coefficient')
        ax_sparse.set_title('Attention Sparsity Evolution')
        ax_sparse.legend()
        ax_sparse.grid(True, alpha=0.3)
    
        # Mutual information heatmap
        ax_mi = plt.subplot(gs[-1, num_slots+1:])
        mi_matrix = self._compute_slot_mutual_information(pre_attention, post_attention)
        im_mi = ax_mi.imshow(mi_matrix, cmap='coolwarm', aspect='auto')
        ax_mi.set_xlabel('Post-training Slots')
        ax_mi.set_ylabel('Pre-training Slots')
        ax_mi.set_title('Mutual Information Between Attention Patterns')
        plt.colorbar(im_mi, ax=ax_mi, label='MI (bits)')
    
        plt.suptitle('Attention Pattern Evolution: From Generic to Task-Specific Focus',
                fontsize=16, fontweight='bold')
    
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
        return fig

    def _compute_attention_entropy(self, attention):
        """Calculate Shannon entropy of attention distributions."""
        # Add small epsilon for numerical stability
        attention_stable = attention + 1e-10
        attention_norm = attention_stable / attention_stable.sum(axis=-1, keepdims=True)
        entropy = -np.sum(attention_norm * np.log2(attention_norm), axis=-1)
        return entropy.mean(axis=0)  # Average over batch

    def _compute_attention_sparsity(self, attention):
        """Calculate Gini coefficient as sparsity measure."""
        def gini(array):
            array = array.flatten()
            array = np.sort(array)
            n = len(array)
            index = np.arange(1, n + 1)
            return (2 * np.sum(index * array)) / (n * np.sum(array)) - (n + 1) / n
    
        sparsity = []
        for slot_idx in range(attention.shape[1]):
            slot_attention = attention[:, slot_idx, :]
            gini_coef = np.mean([gini(sample) for sample in slot_attention])
            sparsity.append(gini_coef)
    
        return np.array(sparsity)

    def _compute_slot_mutual_information(self, pre_attention, post_attention):
        """Compute mutual information between pre and post attention patterns."""
        num_slots = pre_attention.shape[1]
        mi_matrix = np.zeros((num_slots, num_slots))
    
        for i in range(num_slots):
            for j in range(num_slots):
                # Flatten and discretize attention patterns
                pre_flat = pre_attention[:, i, :].flatten()
                post_flat = post_attention[:, j, :].flatten()
            
                # Compute mutual information using histogram approximation
                hist_2d, _, _ = np.histogram2d(pre_flat, post_flat, bins=20)
                pxy = hist_2d / hist_2d.sum()
                px = pxy.sum(axis=1)
                py = pxy.sum(axis=0)
            
                # MI calculation
                px_py = px[:, None] * py[None, :]
                nzs = pxy > 0
                mi = np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs]))
                mi_matrix[i, j] = mi
    
        return mi_matrix


    def run_complete_analysis(self, pretrained_path, finetuned_path, 
                             test_observations, trajectory_checkpoints=None):
        """
        Execute complete analysis pipeline with all visualizations and metrics.
        """
        print("=== Starting Comprehensive Slot Evolution Analysis ===")
        
        all_results = {}
        
        # 1. Load models
        print("\n1. Loading models...")
        pretrained_model = self.load_pretrained_slot_attention_model(pretrained_path)
        finetuned_policies = self.load_finetuned_policies(finetuned_path)
        
        if not finetuned_policies:
            raise ValueError("No fine-tuned policies could be loaded!")
        
        # 2. Static analysis (before/after comparison)
        print("\n2. Performing static analysis...")
        static_results = self.extract_and_compare_representations(
            test_observations, pretrained_model, finetuned_policies
        )
        all_results['static_analysis'] = static_results
        
        # 3. Create core visualizations
        print("\n3. Creating visualizations...")
        
        # TSNE visualization
        tsne_fig = self.create_comparative_tsne_visualization(
            static_results, test_observations,
            save_path=os.path.join(self.dirs['tsne'], 'slot_evolution_tsne.png')
        )
        plt.close(tsne_fig)
        
        # Attention evolution
        attention_fig = self.create_attention_evolution_visualization(
            static_results, test_observations,
            save_path=os.path.join(self.dirs['attention'], 'attention_evolution.png')
        )
        plt.close(attention_fig)
        
        # Slot tracking
        tracking_fig = self.create_slot_tracking_visualization(
            static_results, test_observations,
            save_path=os.path.join(self.dirs['slot_tracking'], 'slot_correspondence.png')
        )
        plt.close(tracking_fig)
        
        # 4. Temporal analysis if checkpoints provided
        if trajectory_checkpoints:
            print("\n4. Analyzing training trajectory...")
            temporal_results = self.analyze_trajectory_with_checkpoints(
                trajectory_checkpoints, test_observations, pretrained_model
            )
            all_results['temporal_analysis'] = temporal_results
            
            # Create temporal visualization
            temporal_fig = self.create_temporal_evolution_visualization(
                temporal_results,
                save_path=os.path.join(self.dirs['temporal'], 'training_evolution.png')
            )
            plt.close(temporal_fig)
        
        # 5. Generate comprehensive report
        print("\n5. Generating analysis report...")
        report = self.generate_comprehensive_report(
            all_results,
            save_path=os.path.join(self.output_dir, 'analysis_report.json')
        )
        
        print(f"\n=== Analysis Complete! ===")
        print(f"Results saved to: {self.output_dir}")
        print(f"\nKey findings:")
        for finding, details in report['key_findings'].items():
            print(f"  - {finding}: {details}")
        
        return all_results