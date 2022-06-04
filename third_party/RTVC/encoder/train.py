from third_party.RTVC.encoder.visualizations import Visualizations
from third_party.RTVC.encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from third_party.RTVC.encoder.params_model import *
from third_party.RTVC.encoder.model import SpeakerEncoder
from third_party.RTVC.utils.profiler import Profiler
from pathlib import Path
from statistics import mean
import torch
import time


def sync(device: torch.device):
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    

def train(run_id: str, clean_data_root: Path, models_dir: Path, umap_every: int, save_every: int,
          backup_every: int, vis_every: int, force_restart: bool, visdom_server: str,
          no_visdom: bool, val_root=None):
    # Create a dataset and a dataloader
    mean_loss = []
    mean_eer = []
    dataset = SpeakerVerificationDataset(clean_data_root)
    loader = SpeakerVerificationDataLoader(
        dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=8
    )
    
    # Setup the device on which to run the forward pass and the loss. These can be different, 
    # because the forward pass is faster on the GPU whereas the loss is often (depending on your
    # hyperparameters) faster on the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = torch.device("cuda")
    
    # Create the model and the optimizer
    model = SpeakerEncoder(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    init_step = 1
    
    # Configure file path for the model
    state_fpath = models_dir.joinpath(run_id + ".pt")
    backup_dir = models_dir.joinpath(run_id + "_backups")

    # Load any existing model
    if not force_restart:
        if state_fpath.exists():
            print("Found existing model \"%s\", loading it and resuming training." % run_id)
            checkpoint = torch.load(state_fpath)
            init_step = checkpoint["step"]
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = learning_rate_init
        else:
            print("No model \"%s\" found, starting training from scratch." % run_id)
    else:
        print("Starting the training from scratch.")
    model.train()
    
    # Initialize the visualization environment
    vis = Visualizations(run_id, vis_every, server=visdom_server, disabled=no_visdom)
    vis.log_dataset(dataset)
    vis.log_params()
    device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    vis.log_implementation({"Device": device_name})
    
    # Training loop
    profiler = Profiler(summarize_every=10, disabled=False)
    for step, speaker_batch in enumerate(loader, init_step):
        if step % 100 == 0:
            print('------------------------------------------')
            print(f'Mean Loss per 100 steps: {mean(mean_loss)}')
            print(f'Mean EER per 100 steps: {mean(mean_eer)}')
            print('------------------------------------------')
            mean_loss.clear()
            mean_eer.clear()
        profiler.tick("Blocking, waiting for batch (threaded)")
        
        # Forward pass
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        sync(device)
        profiler.tick("Data to %s" % device)
        embeds = model(inputs)
        sync(device)
        profiler.tick("Forward pass")
        embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
        loss, eer = model.loss(embeds_loss)
        sync(loss_device)
        profiler.tick("Loss")

        # Backward pass
        model.zero_grad()
        loss.backward()
        profiler.tick("Backward pass")
        model.do_gradient_ops()
        optimizer.step()
        profiler.tick("Parameter update")
        
        # Update visualizations
        # learning_rate = optimizer.param_groups[0]["lr"]
        vis.update(loss.item(), eer, step)
        mean_loss.append(loss.item())
        mean_eer.append(eer)

        
        # Draw projections and save them to the backup folder
        if umap_every != 0 and step % umap_every == 0:
            print("Drawing and saving projections (step %d)" % step)
            backup_dir.mkdir(exist_ok=True)
            projection_fpath = backup_dir.joinpath("%s_umap_%06d.png" % (run_id, step))
            embeds = embeds.detach().cpu().numpy()
            vis.draw_projections(embeds, utterances_per_speaker, step, projection_fpath)
            vis.save()

        # Overwrite the latest version of the model
        if save_every != 0 and step % save_every == 0:
            print("Saving the model (step %d)" % step)
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, state_fpath)
            
        # Make a backup
        if backup_every != 0 and step % backup_every == 0:
            print("Making a backup (step %d)" % step)
            backup_dir.mkdir(exist_ok=True)
            backup_fpath = backup_dir.joinpath("%s_bak_%06d.pt" % (run_id, step))
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, backup_fpath)
            
        profiler.tick("Extras (visualizations, saving)")
