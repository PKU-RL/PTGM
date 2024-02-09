import os
from steve1.utils.file_utils import load_pickle

class Steve1Codebook:
    def __init__(self, path='downloads/steve1/visual_prompt_embeds'):
        fs = os.listdir(path)
        self.N = 0
        self.codebook = []
        for f in fs:
            #print(f)
            self.codebook.append(load_pickle(os.path.join(path, f)))
            self.N += 1
        #print(self.codebook)

    def get_code(self, i):
        return self.codebook[i]

class KMeansCodebook:
    def __init__(self, path='downloads/centers.pkl'):
        self.codebook = load_pickle(path)
        self.N = len(self.codebook)
        #print(self.codebook)

    def get_code(self, i):
        return self.codebook[i].astype('float32')