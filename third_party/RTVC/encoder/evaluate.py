from third_party.RTVC.encoder.visualizations import Visualizations
from third_party.RTVC.encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from third_party.RTVC.encoder.params_model import *
from third_party.RTVC.encoder.model import SpeakerEncoder
from third_party.RTVC.utils.profiler import Profiler
from pathlib import Path
import torch
import umap
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def sync(device: torch.device):
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(run_id: str,  models_dir: Path, val_root: Path, num_iters: int):
    val_data = SpeakerVerificationDataset(val_root)
    val_loader = SpeakerVerificationDataLoader(
        val_data,
        val_speakers_per_batch,
        val_utterances_per_speaker,
        num_workers=6
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # FIXME: currently, the gradient is None if loss_device is cuda
    loss_device = torch.device("cpu")

    # Create the model and the optimizer
    model = SpeakerEncoder(device, loss_device)
    state_fpath = models_dir.joinpath(run_id + ".pt")
    backup_dir = models_dir.joinpath(run_id + "val_backups")

    if state_fpath.exists():
        print("Found existing model \"%s\", loading it." % run_id)
        checkpoint = torch.load(state_fpath)
        init_step = checkpoint["step"]
        print(f'From {init_step} steps')
        model.load_state_dict(checkpoint["model_state"])
        # optimizer.load_state_dict(checkpoint["optimizer_state"])
        # optimizer.param_groups[0]["lr"] = learning_rate_init
    else:
        print("No model \"%s\" found." % run_id)

    model.eval()

    iters = 0
    mean_loss = 0
    mean_eer = 0

    with torch.no_grad():

        for i, val_batch in enumerate(val_loader):
            # for speaker in val_batch.speakers:
            #     for ut in speaker.utterance_cycler.all_items:
            #         print(ut.wave_fpath)
            if iters > num_iters:
                break

            inputs = torch.from_numpy(val_batch.data).to(device)
            sync(device)
            embeds = model(inputs)
            sync(device)
            embeds_loss = embeds.view((val_speakers_per_batch, val_utterances_per_speaker, -1)).to(loss_device)
            loss, eer = model.loss(embeds_loss)
            sync(loss_device)
            # print(f'{i} step val_loss: {loss.item()}')
            # print(f'{i} step val_eer: {eer}')
            if i % 10 == 0:
                print(f'{i} steps from start')

            # print("Drawing and saving projections (step %d)" % i)
            # backup_dir.mkdir(exist_ok=True)
            # projection_fpath = backup_dir.joinpath("%s_umap_%06d.png" % (run_id, i))
            # embeds = embeds.detach().cpu().numpy()
            # draw_projections(embeds, val_utterances_per_speaker, i, projection_fpath, max_speakers=10)

            mean_loss += loss.item()
            mean_eer += eer
            iters += 1

        print('----------------')
        print(f'Mean loss: {mean_loss/(num_iters+1)}')
        print(f'Mean eer: {mean_eer/(num_iters+1)}')


def draw_projections(embeds, utterances_per_speaker, step, out_fpath=None,
                     max_speakers=10):
    max_speakers = min(max_speakers, len(colormap))
    embeds = embeds[:max_speakers * utterances_per_speaker]

    n_speakers = len(embeds) // utterances_per_speaker
    ground_truth = np.repeat(np.arange(n_speakers), utterances_per_speaker)
    colors = [colormap[i] for i in ground_truth]

    reducer = umap.UMAP()
    projected = reducer.fit_transform(embeds)
    plt.scatter(projected[:, 0], projected[:, 1], c=colors)
    plt.gca().set_aspect("equal", "datalim")
    plt.title("UMAP projection (step %d)" % step)
    if out_fpath is not None:
        plt.savefig(out_fpath)
    plt.clf()




colormap = np.array([
    [76, 255, 0],
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
], dtype=np.float) / 255