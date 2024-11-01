import numpy as np
import torch

class EVSProcessing:
    def __init__(self, file_dir, file_name, size):
        self.file_dir = file_dir
        self.file_name = file_name
        self.size = size

    def evs_process(self, evs):
        evs[:, 0] = evs[:, 0] - evs[0][0] + 1
        t_norm = max(evs[:, 0])
        evs = np.c_[evs, np.where(evs[:, -1] == 0, -1, evs[:, -1]), evs[:, 0] / t_norm]
        return evs[:, [0, 2, 1, 3, 4, 5]], t_norm

    def evs2dict(self, evs):
        evs_dict = {}
        for i in evs:
            key = (i[1], i[2])
            if key not in evs_dict:
                evs_dict[key] = [[], []]
            evs_dict[key][0].append(i[-3])
            evs_dict[key][1].append(i[-1])
        return evs_dict

    def depth_from_dict(self, evs_dict):
        max_d, temp_k = 0, None
        for k, v in evs_dict.items():
            if len(v[0]) > max_d:
                max_d, temp_k = len(v[0]), k
        return max_d, temp_k

    def depth_representation(self, evs_dict, max_d):
        depth_img = []
        for i in range(1, max_d + 1):
            img = np.zeros(self.size)
            img.fill(0.5)
            for k, v in evs_dict.items():
                img[int(k[0]), int(k[1])] = v[0][i - 1] if len(v[0]) >= i else 0.5
            depth_img.append(img.reshape(-1, *img.shape))
        return np.transpose(np.transpose(np.asarray(depth_img), (1,0,2,3))[0], (1,2,0))

    def temporal_inout(self, evs_dict, max_d):
        indata, outdata = [], []
        for k, v in evs_dict.items():
            inarr = np.zeros(len(k) + max_d)
            outarr = np.zeros(len(k) + max_d)
            inarr[:2] = outarr[:2] = k[0] / self.size[0], k[1] / self.size[1]
            dep = min(max_d, len(v[0]))
            inarr[2:2 + dep] = [v[0][i] if v[0][i] == 1 else 0.5 for i in range(dep)]
            outarr[2:2 + dep] = v[1][:dep]
            indata.append(inarr)
            outdata.append(outarr)
        return np.asarray(indata), np.asarray(outdata)[:, 2:]

    def prepare_data(self):
        evs, t_norm = self.evs_process(np.load(f"{self.file_dir}/{self.file_name}"))
        evs_dict = self.evs2dict(evs)
        max_d, _ = self.depth_from_dict(evs_dict)
        evs_voxel = self.depth_representation(evs_dict, max_d)
        indata, outdata = self.temporal_inout(evs_dict, max_d)
        print('Max Timestamp = ', t_norm)
        print('Max Size = ', self.size[0], self.size[1])
        return evs_voxel, (indata, outdata), evs

    def value_clean(self, evs_sr, low=0.25, high=0.75):
        h, w, max_d = evs_sr.shape
        depth_img = []
        for i in range(max_d):
            c = evs_sr[:, :, i].reshape(-1, 1)
            c = np.clip(c, 0, 1) if (c.max() != 1 or c.min() != 0) else c
            c[c <= low], c[c >= high] = 0, 1
            c[(c > low) & (c < high)] = 0.5
            depth_img.append(c.reshape(-1, h, w))
        return np.transpose(np.transpose(np.asarray(depth_img), (1,0,2,3))[0], (1,2,0))

    def sr2indata(self, evs_sr):
        h, w, max_d = evs_sr.shape
        indata = []
        for i in range(h):
            for j in range(w):
                temp = evs_sr[i, j, :].copy()
                if (temp == np.array([0.5] * max_d)).all(): continue
                temp[temp < 1] = abs(temp[temp < 1] - 0.5)
                arr = np.concatenate((np.array([i / h, j / w]), temp), axis=None)
                indata.append(arr)
        return np.asarray(indata)

    def voxel2evs(self, evs_sr, low=0.25, high=0.75):
        return self.sr2indata(self.value_clean(evs_sr, low, high))

    def sr_evs_shape(self, indata_sr, outdata_sr, sf):
        if len(indata_sr) != len(outdata_sr): return
        out = []
        for idxi, i in enumerate(indata_sr):
            for idxj, j in enumerate(i[2:]):
                if j == 0: continue
                out.append([i[0], i[1], j, outdata_sr[idxi][idxj]])
        out = np.asarray(out)
        out[:, 0] = np.around(out[:, 0] * self.size[0] * sf[-1][0])
        out[:, 1] = np.around(out[:, 1] * self.size[1] * sf[-1][1])
        out[:, 2] = np.ceil(out[:, 2] - 0.5)
        return out[np.lexsort((out[:,1], out[:,0])) ]

    def refine_sr_evs(self, sr_evs, refine=0, range_n=1):
        if refine == 0: 
            sr_evs = sr_evs[sr_evs[:, -1] != 0]
            return sr_evs[sr_evs[:, -1].argsort()]
        refine_evs = sr_evs[sr_evs[:, -1] == 0].copy()
        for i in range(len(refine_evs)):
            nb = sr_evs[np.where((sr_evs[:, 0] >= refine_evs[i][0] - range_n) & 
                                 (sr_evs[:, 0] <= refine_evs[i][0] + range_n) & 
                                 (sr_evs[:, 1] >= refine_evs[i][1] - range_n) & 
                                 (sr_evs[:, 1] <= refine_evs[i][1] + range_n))]
            if len(nb) == 0: refine_evs[i][-1] = 0
            else: refine_evs[i][-1] = np.median(nb[:,-1])
        sr_evs = np.vstack((sr_evs, refine_evs))
        sr_evs = sr_evs[sr_evs[:,-1] != 0]
        return sr_evs[sr_evs[:, -1].argsort()]

    def sr_evs_reshape(self, indata_sr, outdata_sr, sf, refine=0, range_n=1):
        return self.refine_sr_evs(self.sr_evs_shape(indata_sr, outdata_sr, sf), 0, range_n)

    def init_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)