import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import zarr
import numpy as np
from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.umi_dataset import UmiDataset
from diffusers import DDIMScheduler
import os
import wandb
from diffusion_policy.dataset.base_dataset import BaseImageDataset

# Configuration parameters from umi.yaml
dataset_path = "dataset.zarr.zip"
batch_size = 64
num_workers = 4
device = "cuda:0" if torch.cuda.is_available() else "cpu"
num_epochs = 120
lr = 3.0e-4
use_ema = True
n_action_steps = 8 # from train_diffusion_unet_timm_umi_workspace.yaml

# from umi.yaml
camera_obs_latency=0.125
robot_obs_latency=0.0001
gripper_obs_latency= 0.02
dataset_frequency=0 #59.94
obs_down_sample_steps= 3

eef_latency= camera_obs_latency - robot_obs_latency*dataset_frequency
gripper_latency= camera_obs_latency - gripper_obs_latency*dataset_frequency

img_obs_horizon = 2 
low_dim_obs_horizon =2 
action_horizon = 16   
ignore_proprioception= False


shape_meta = {
    'obs': {
        'camera0_rgb': {'shape': (3, 224, 224), 
                        'horizon': img_obs_horizon, 
                        'latency_steps': 0, 
                        'down_sample_steps': obs_down_sample_steps, 
                        'type': 'rgb', 
                        'ignore_by_policy': False},

        'robot0_eef_pos': {'shape': (3, ), 
                            'horizon': low_dim_obs_horizon, 
                            'latency_steps': eef_latency, 
                            'down_sample_steps': obs_down_sample_steps, 
                            'type': 'low_dim', 
                            'ignore_by_policy': ignore_proprioception},

        'robot0_eef_rot_axis_angle': {'raw_shape': (3,), 
                                        'shape': (6,), 
                                        'horizon': low_dim_obs_horizon, 
                                        'latency_steps': eef_latency, 
                                        'down_sample_steps': obs_down_sample_steps, 
                                        'type': 'low_dim', 
                                        'rotation_rep': 'rotation_6d', 
                                        'ignore_by_policy': ignore_proprioception},

        'robot0_gripper_width': {'shape': (1,), 
                                    'horizon': low_dim_obs_horizon, 
                                    'latency_steps': gripper_latency, 
                                    'down_sample_steps': obs_down_sample_steps, 
                                    'type': 'low_dim', 
                                    'ignore_by_policy': ignore_proprioception},

        'robot0_eef_rot_axis_angle_wrt_start': {'raw_shape': (3,), 
                                                    'shape': (6,), 
                                                    'horizon': low_dim_obs_horizon, 
                                                    'latency_steps': eef_latency, 
                                                    'down_sample_steps': obs_down_sample_steps, 
                                                    'type': 'low_dim', 
                                                    'ignore_by_policy': ignore_proprioception}
    },
    'action': {
        'shape': (10,),
        'horizon': action_horizon,
        'latency_steps': 0,
        'down_sample_steps': obs_down_sample_steps,
        'rotation_rep': 'rotation_6d',
    }
}

# obs_keys = [
#     'camera0_rgb',
#     'robot0_eef_pos',
#     'robot0_eef_rot_axis_angle',
#     'robot0_gripper_width',
#     'robot0_eef_rot_axis_angle_wrt_start'
# ]

pose_repr = {'obs_pose_repr': 'relative', 'action_pose_repr': 'relative'}

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load dataset
train_dataset = UmiDataset(shape_meta=shape_meta, 
                        dataset_path=dataset_path,
                        cache_dir=None,
                        pose_repr=pose_repr,
                        action_padding=False,
                        temporally_independent_normalization=False,
                        repeat_frame_prob=0.0,
                        seed=42,
                        val_ratio=0.05,
                        )
val_dataset = train_dataset.get_validation_dataset()

# Create dataloaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers= True #added
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers= True #added
)

# Initialize observation encoder
obs_encoder = TimmObsEncoder(
    shape_meta=shape_meta,
    model_name='vit_base_patch16_clip_224.openai',
    pretrained=True,
    frozen=False,
    global_pool='',
    feature_aggregation= 'attention_pool_2d',
    position_encording='sinusoidal',
    downsample_ratio= 32,
    transforms= [{'type':'RandomCrop', 'ratio': 0.95}],
    use_group_norm=True,
    share_rgb_model=False,
    imagenet_norm= True
)

# Initialize noise scheduler
noise_scheduler = DDIMScheduler(
    num_train_timesteps=50,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    set_alpha_to_one=True,
    steps_offset=0,
    prediction_type='epsilon'
)


# Initialize diffusion policy
policy = DiffusionUnetTimmPolicy(
    shape_meta=shape_meta,
    noise_scheduler=noise_scheduler,
    obs_encoder=obs_encoder,
    num_inference_steps=16,
    obs_as_global_cond=True,
    diffusion_step_embed_dim=128,
    down_dims=[256, 512, 1024],
    kernel_size=5,
    n_groups=8,
    cond_predict_scale=True,
    input_pertub=0.1,
    train_diffusion_n_samples=1
)

# Set normalizer from dataset
policy.set_normalizer(train_dataset.get_normalizer())

# Move policy to device
policy.to(device)

# Initialize EMA if enabled
if use_ema:
    ema_model = EMAModel(model=policy,
                        update_after_step= 0,
                        inv_gamma= 1.0,
                        power= 0.75,
                        min_value= 0.0,
                        max_value= 0.9999)

# Initialize optimizer
optimizer = optim.AdamW(
    policy.parameters(),
    lr=lr,
    betas=(0.95, 0.999),
    eps=1.0e-8,
    weight_decay=1.0e-6
)

# Initialize Weights & Biases
wandb.init(project="diffusion_policy_training", name="train_diffusion_unet_timm")

# Create checkpoint directory
os.makedirs("checkpoints", exist_ok=True)

# Training loop
for epoch in range(num_epochs):
    policy.train()
    train_losses = []
    for batch in train_dataloader:
        batch = {k: v.to(device) if k != 'obs' else {ok: ov.to(device) for ok, ov in v.items()} 
                 for k, v in batch.items()}
        loss = policy.compute_loss(batch)
        # loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if use_ema:
            ema_model.step(policy)
        train_losses.append(loss.item())
    
    train_loss = np.mean(train_losses)
    wandb.log({"train_loss": train_loss, "epoch": epoch})

    # Validation
    policy.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) if k != 'obs' else {ok: ov.to(device) for ok, ov in v.items()} 
                     for k, v in batch.items()}
            loss = policy.compute_loss(batch)
            val_losses.append(loss.item())
    val_loss = np.mean(val_losses)
    wandb.log({"val_loss": val_loss, "epoch": epoch})

    # Save checkpoint every 10 epochs
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        checkpoint_path = os.path.join("checkpoints", f"checkpoint_epoch_{epoch}.ckpt")
        torch.save(policy.state_dict(), checkpoint_path)
        print(f"Saved checkpoint at epoch {epoch} to {checkpoint_path}")

# Save final model
final_model_path = os.path.join("checkpoints", "final_model.ckpt")
torch.save(policy.state_dict(), final_model_path)
print(f"Saved final model to {final_model_path}")

# Finish Weights & Biases run
wandb.finish()