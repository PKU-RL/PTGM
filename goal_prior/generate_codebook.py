from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import random
import os, imageio, sys, pickle
from scipy.spatial.distance import euclidean
sys.path.append(os.path.abspath(''))
from steve1.data.EpisodeStorage import EpisodeStorage
from steve1.data.minecraft_dataset import load_sampling
import argparse
from tqdm import tqdm

def plot_embedding(data, y, save_path, centers=None):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    #ax = plt.subplot(111)
    plt.scatter(data[:, 0], data[:, 1], c=y)
    
    if centers is not None:
        centers = (centers - x_min) / (x_max - x_min)
        plt.scatter(centers[:, 0], centers[:, 1], c='r')
    plt.savefig(os.path.join(save_path, 'tsne.png'))

'''
def save_frame_labels(frames, labels, path):
    for i, (f, l) in enumerate(zip(frames, labels)):
        p = os.path.join(path, str(l))
        if not os.path.exists(p):
            os.mkdir(p)
        imageio.imsave(os.path.join(p, '{}.png'.format(i)), f)
'''

def save_center_video(paths, steps, codes, center_idxs, output_dir):
    '''
    save the codebook list, and the corresponding 16 frames of each code
    '''
    cts = []
    for _, i in enumerate(center_idxs):
        cts.append(codes[i].reshape((1,-1)))
        episode = EpisodeStorage(paths[i])
        frames = episode.load_frames(only_range=(steps[i], steps[i]+16))
        imageio.mimsave(os.path.join(output_dir, 'center_{}.gif'.format(_)), 
            frames[steps[i]:steps[i]+16], 'GIF', duration=0.05)
    #print(cts)
    with open(os.path.join(output_dir, 'centers.pkl'), 'wb') as f:
        pickle.dump(cts, f)
        #pickle.close()

def dict2array(d):
    ret = []
    for k in d:
        if hasattr(d[k], "__len__"):
            ret += list(d[k])
        else:
            ret.append(d[k])
    #print(d, ret)
    return np.asarray(ret, dtype=float)



def sample_embeddings(files, history_act='add', interval=10):
    '''
    sample goal embeddings from the video dataset, for clustering
    params:
        files: dataset path list
        history_act: how to aggregate last 16 actions to concate with visual embeddings
            add action information can make different behaviors more separable
        interval: how many steps to preserve a goal
    return:
        paths: list of video path
        steps: list of selected index in the video
        codes: list of MineCLIP embedding (for codebook)
        embeds: list of [MineCLIP embedding, actions] (for clustering)
    '''
    act_dim=25 if history_act=='add' else 400
    paths, steps, codes, embeds = [], [], np.empty(shape=(0,512)), np.empty(shape=(0,512+act_dim))

    for f in tqdm(files):
        episode = EpisodeStorage(f)
        es = np.asarray(episode.load_embeds_attn()[15:]).reshape((-1,512))
        #fs = episode.load_frames()[15:]
        pths = [f] * es.shape[0]
        stps = [i for i in range(es.shape[0])]
        acts = episode.load_actions()
        acts = np.asarray([dict2array(a) for a in acts])    
        acts = [np.roll(acts, i, axis=0) for i in range(0,16)] # concat last 16 frames actions
        if history_act=='add':
            acts = np.sum(acts, axis=0)[15:]
        else:
            acts = np.concatenate(acts, axis=1)[15:]
        cs = es
        es = np.concatenate((es, acts), axis=1)
        #frames += fs 
        paths += pths 
        steps += stps
        codes = np.concatenate((codes, cs))
        embeds = np.concatenate((embeds, es))
    embeds = np.asarray(embeds).reshape((-1,512+act_dim))
    codes = np.asarray(codes).reshape((-1,512))
    # preserve a small amount of embeddings with interval
    # because nearby embeddings are very similar. preserving all embeddings makes kmeans very slow
    sample_idxs = np.arange(0, len(paths), interval)
    paths = np.asarray(paths)[sample_idxs]
    steps = np.asarray(steps)[sample_idxs]
    codes = codes[sample_idxs]
    embeds = embeds[sample_idxs]
    print('Loaded {} frames in total'.format(len(paths)))
    #print(len(paths), len(steps), codes.shape, embeds.shape)
    return paths, steps, codes, embeds


def main(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    exp_name = '{}_n{}_s{}'.format(args.embedding, args.n_codebook, args.seed)
    output_dir = os.path.join(args.output_dir, exp_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)

    data = load_sampling(args.sampling_dir, args.sampling)[0]
    data = random.sample(data, args.video_num)
    print(data)

    paths, steps, codes, embeds = sample_embeddings(data, interval=args.sample_interval)

    if args.embedding == 'tsne':
        tsne = TSNE(n_components=2)
        result = tsne.fit_transform(embeds)
    else:
        raise NotImplementedError

    print('running kmeans clustering')
    kmeans = KMeans(n_clusters=args.n_codebook, init='k-means++', random_state=args.seed).fit(result)
    y_pred = kmeans.labels_
    centers = kmeans.cluster_centers_
    center_idxs = []
    print('done')

    for iclust in range(kmeans.n_clusters):
        cluster_pts = result[kmeans.labels_ == iclust]
        # get all indices of points assigned to this cluster:
        cluster_pts_indices = np.where(kmeans.labels_ == iclust)[0]
        cluster_cen = centers[iclust]
        min_idx = np.argmin([euclidean(result[idx], cluster_cen) for idx in cluster_pts_indices])
        idx = cluster_pts_indices[min_idx]
        #print(idx, cluster_cen, result[idx])
        center_idxs.append(idx)

    plot_embedding(result, y_pred, output_dir, result[center_idxs])
    save_center_video(paths, steps, codes, center_idxs, output_dir)
    #save_frame_labels(frames, y_pred, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', type=str, default='tnse')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output-dir', type=str, default='downloads/codebook/')
    parser.add_argument('--sampling_dir', type=str, default='downloads/samplings/')
    parser.add_argument('--sampling', type=str, default='seed1')
    parser.add_argument('--video-num', type=int, default=50) # number of videos to sample goals
    parser.add_argument('--sample-interval', type=int, default=50) # for each video, sample video_len/interval goals
    parser.add_argument('--n-codebook', type=int, default=100)

    args = parser.parse_args()
    main(args)
