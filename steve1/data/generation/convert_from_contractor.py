import argparse
import os, sys
import numpy as np
import cv2
import torch
from tqdm import tqdm

#print(sys.path)
#print(os.path.abspath(''))
sys.path.append(os.path.abspath(''))

from steve1.config import MINECLIP_CONFIG
from steve1.data.contractor import ContractorData
from steve1.data.EpisodeStorage import EpisodeStorage
from steve1.utils.load_mineclip import load
from steve1.data.generation.FrameBuffer import QueueFrameBuffer
#from steve1.data.generation.gen_mixed_agents import embed_videos_mineclip_batched

@torch.no_grad()
def embed_videos_mineclip_batched(frame_buffer: QueueFrameBuffer, mineclip, device, batch_size=32):
    """Compute mineclip_code video embedding for an entire QueueFrameBuffer. Returns a listr of 512 vectors
    with shape (1, 512).
    """
    print(f'Embedding {len(frame_buffer)} frames with batch size {batch_size}...')

    frame_iter = iter(frame_buffer)
    prog = tqdm(total=len(frame_buffer))
    video_embeds_all = []
    done = False
    while not done:
        # Get batch of videos
        videos = []
        for _ in range(batch_size):
            try:
                frame = next(frame_iter)
                prog.update(1)
            except StopIteration:
                done = True
                break
            videos.append(frame)
        if len(videos) == 0:
            break

        # Compute embeddings in batched form
        video_batch = torch.cat(videos).to(device)
        bsz = video_batch.shape[0]
        # Autocast so that we can use fp16
        with torch.cuda.amp.autocast():
            video_embeds = mineclip.encode_video(video_batch)
        video_embeds = video_embeds.detach().cpu().numpy()
        assert video_embeds.shape == (bsz, 512)  # batch of 512-vectors

        # Add to list (each embed is its own element)
        for video_embed in video_embeds:
            video_embed = video_embed.reshape(1, 512)
            assert video_embed.shape == (1, 512)
            video_embeds_all.append(video_embed)
    return video_embeds_all


def convert_episode(contractor_data, idx, mineclip, ep_dirpath, batch_size, min_timesteps):
    print(f'Downloading and converting episode {idx}...')
    try:
        frames, frames_mineclip, actions = contractor_data.download(idx)
        if frames is None:
            print(f'Episode {idx} not valid. Skipping...')
            return
        num_timesteps = len(frames)
        if num_timesteps < min_timesteps:
            print(f'Episode has {num_timesteps} timesteps, less than {min_timesteps}. Skipping...')
            return
        frame_buffer = QueueFrameBuffer()
        for frame in frames_mineclip:
            frame_buffer.add_frame(frame)
        print(f'Embedding frames...')
        mineclip_embeds = embed_videos_mineclip_batched(frame_buffer, mineclip, 'cuda', batch_size)
        mineclip_embeds = [None] * 15 + mineclip_embeds
        ep = EpisodeStorage(ep_dirpath)
        for frame, action, embed in zip(frames, actions, mineclip_embeds):
            ep.append(frame, action, embed)

        metadata = {
            'contractor_version': contractor_data.version,
            'contractor_index': idx,
        }

        labeled_episode_dirpath = ep_dirpath + f'_contractor_nmFrm{num_timesteps}_term'
        ep.update_episode_dirpath(labeled_episode_dirpath)
        
        ep.save_episode()
        ep.save_metadata(metadata)
        print(f'Episode saved to {labeled_episode_dirpath}')
    except Exception as e:
        print(f'Error processing episode {idx}: {e}')
        return

def episode_exists(ep_dirpath, existing_episodes):
    ep_dirpath = ep_dirpath.split('/')[-1]
    for existing_ep in existing_episodes:
        existing_ep = existing_ep.split('_contractor_nmFrm')[0]
        if ep_dirpath == existing_ep:
            return True
    return False

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Loading MineCLIP...')
    mineclip = load(MINECLIP_CONFIG, device='cuda')
    print(f'Loading contractor data index {args.index}...')
    contractor_data = ContractorData(args.index, args.cache_dir)
    print(f'Index loaded.')
    print(f'This dataset has {len(contractor_data)} episodes.')
    index_name = args.index.replace('.', '_')
    start_idx = args.worker_id * args.num_episodes
    end_idx = start_idx + args.num_episodes
    # option to download all the data
    if args.worker_id==0 and args.num_episodes==-1:
        start_idx = args.start_idx
        end_idx = len(contractor_data)
    print('Download idxs: {}~{}'.format(start_idx,end_idx))
    # Get all files in output_dir
    existing_episodes = os.listdir(args.output_dir)
    for idx in range(start_idx, end_idx):
        print(f'Converting episode {idx}...')
        episode_name = f'contractor_{index_name}_{idx}'
        ep_dirpath = os.path.join(args.output_dir, episode_name)
        if episode_exists(ep_dirpath, existing_episodes):
            print(f'Episode {idx} already exists at {ep_dirpath}. Skipping...')
            continue
        convert_episode(contractor_data, 
                        idx, 
                        mineclip, 
                        ep_dirpath, 
                        args.batch_size,
                        args.min_timesteps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, default='10.x', choices=['6.x',
                                                                      '7.x',
                                                                      '8.x',
                                                                      '9.x',
                                                                      '10.x'])
    parser.add_argument('--output_dir', type=str, default='downloads/dataset_contractor/')
    parser.add_argument('--cache_dir', type=str, default='downloads/contractor_cache')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--worker_id', type=int, default=0)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=200)
    parser.add_argument('--min_timesteps', type=int, default=1000)

    args = parser.parse_args()
    print('Download dataset {}, start idx {}'.format(args.index, args.start_idx))
    main(args)