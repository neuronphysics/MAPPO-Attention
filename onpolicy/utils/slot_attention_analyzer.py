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
            share_observation_space = self.envs.share_observation_space[player_key] if self.use_centralized_V else \
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