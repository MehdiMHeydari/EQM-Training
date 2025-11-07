from torch.utils.data import Dataset
import numpy as np
from einops import rearrange
from math import floor

class VF_FM(Dataset):
    def __init__(self, data, all_vel=True) -> None:
        super().__init__()
        self.all_vel = all_vel
        self._preprocess(data, 'data')
        
        if self.data.ndim == 4:
            self.shape = self.data.shape[1:]
            self.one_yp = True
            self.wm_vf = False
        elif self.data.ndim == 5:
            if self.all_vel == True:
                self.wm_vf = False
                self.shape = self.data.shape[2:]
                self.num_yp = self.data.shape[1]
            else:
                self.wm_vf = True
            self.one_yp = False
        else:
            raise ValueError("Check the members of the dataset!")
                
    def _preprocess(self, data, name):
        setattr(self, name, (data).astype(np.float32))
               
    def __len__(self):
        if self.one_yp or self.wm_vf:  
            return self.data.shape[0]
        else:
            return self.data.shape[0]*self.data.shape[1]
    
    def __getitem__(self, index):
        if self.one_yp and not self.wm_vf:
            return  np.empty(self.shape, dtype=np.float32), self.data[index]
        
        elif not self.one_yp and not self.wm_vf:
            yp_ind = index % self.num_yp
            batch = index // self.num_yp
            return np.empty(self.shape, dtype=np.float32), self.data[batch, yp_ind], yp_ind
        
        else:
            return self.data[index, 0], self.data[index, 1]
        
class VF_FM_patchify(VF_FM):
    def __init__(self, data, all_vel, patch_dims):
       super().__init__(data, all_vel)
       self.patch_dims = patch_dims
       self.space_ress = self.data.shape[-2:]
               
    def __getitem__(self, index):
        start_indices = [np.random.randint(0,space_res-patch_dim) if space_res > patch_dim else 0 for space_res, patch_dim in zip(self.space_ress, self.patch_dims)]
        patch_x, patch_y = (slice(start_index, start_index+patch_dim) for start_index, patch_dim in zip(start_indices,self.patch_dims))
        if not self.one_yp and not self.wm_vf:
            x0, x1, y = super().__getitem__(index)
            return x0[..., patch_x, patch_y], x1[..., patch_x, patch_y], y
        else:
            x0, x1 = super().__getitem__(index)
            return x0[..., patch_x, patch_y], x1[..., patch_x, patch_y]
    
class Joint(Dataset):
    def __init__(self, data, contrastive=False):
        super().__init__()
        self.data = data
        self.shape = data.shape[1:]
        self.contrastive = contrastive
        self.n = data.shape[0]
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        v1, v2 =  np.empty(self.shape, dtype=np.float32), self.data[index]
        if self.contrastive:
            random_index = np.random.randint(0, self.n)
            while index == random_index:
                random_index = np.random.randint(0, self.n)
            v1_cont, v2_cont = np.empty(self.shape, dtype=np.float32), self.data[random_index]
            return v1, v2, v1_cont, v2_cont
        return v1, v2
    
# class Wallpres(Dataset):
#     def __init__(self, data, mean, std) -> None:
#         super().__init__()
#         data = rearrange(data, "b l c w -> (b c) l w")
#         data = (data[:, :160])[:, None]
#         self.data = (data).astype(np.float32)
#         self._norm(mean, std)
#         self.shape = self.data.shape[1:]
        
#     def _norm(self, mean, std):
#         self.data = (self.data - mean)/std

#     def __len__(self):
#         return self.data.shape[0]
    
#     def __getitem__(self, index):
#         return np.empty(self.shape, dtype=np.float32), self.data[index]
    
# class KS(Dataset):
#     def __init__(self, data, mean, std) -> None:
#         super().__init__()
#         data = data[:, None]
#         self.data = (data).astype(np.float32)
#         self._norm(mean, std)
#         self.shape = self.data.shape[1:]
        
#     def _norm(self, mean, std):
#         self.data = (self.data - mean)/std

#     def __len__(self):
#         return self.data.shape[0]
    
#     def __getitem__(self, index):
#         return np.empty(self.shape, dtype=np.float32), self.data[index]
    
# class WPWS(Dataset):
#     def __init__(self, wp, ws, mean_wp, std_wp, mean_ws, std_ws, cutoff=160) -> None:
#         super().__init__()
#         self._preprocess(wp, cutoff, 'wp')
#         self._norm(mean_wp, std_wp, 'wp')
#         self._preprocess(ws, cutoff,'ws')
#         self._norm(mean_ws, std_ws, 'ws')       

#     def _preprocess(self, data, cutoff, name):
#         data = rearrange(data, "b l c w -> (b c) l w")
#         data = (data[:, :cutoff])[:, None] if cutoff else data[:, None]
#         setattr(self, name, (data).astype(np.float32))
        
#     def _norm(self, mean, std, name):
#         data = getattr(self, name)
#         data = (data - mean)/std
#         setattr(self, name, data)
               
#     def __len__(self):  
#         return self.wp.shape[0]
    
#     def __getitem__(self, index):
#         return self.wp[index], self.ws[index]
    
# class WPWS_DD(Dataset):
#     def __init__(self, wp, ws, mean_wp, std_wp, mean_ws, std_ws, cutoff=160, uniform_prob=None, uniform_probs_list=None) -> None:
#         super().__init__()
#         assert uniform_prob is None or uniform_probs_list is None, "Either uniform prob or uniform prob list should be None"
#         if uniform_prob is not None:
#             self.uniform_prob = uniform_prob
#             self.uniform_probs_list = None
#         elif uniform_probs_list is not None:
#             self.uniform_probs_list = uniform_probs_list
#             self.uniform_prob = None
#         self._preprocess(wp, cutoff, 'wp')
#         self._norm(mean_wp, std_wp, 'wp')
#         self._preprocess(ws, cutoff,'ws')
#         self._norm(mean_ws, std_ws, 'ws')       
#         self.mask_shape = (self.wp[0]).shape 

#     def _preprocess(self, data, cutoff, name):
#         data = rearrange(data, "b l c w -> (b c) l w")
#         data = (data[:, :cutoff])[:, None] if cutoff else data[:, None]
#         setattr(self, name, (data).astype(np.float32))
        
#     def _norm(self, mean, std, name):
#         data = getattr(self, name)
#         data = (data - mean)/std
#         setattr(self, name, data)
               
#     def __len__(self):  
#         return self.wp.shape[0]
    
#     def __getitem__(self, index):
#         prob = np.random.choice(self.uniform_probs_list) if self.uniform_probs_list is not None else self.uniform_prob
#         mask = np.random.choice([0, 1], size=self.mask_shape, p=[prob, 1.-prob]).astype(np.float32)
#         return np.concatenate((self.wp[index]*mask, mask)), np.concatenate((self.ws[index]*mask, mask)) # self.wp[index]*mask, self.ws[index]
    
# class WMVF(Dataset):
#     def __init__(self, wp, ws_u, ws_w, 
#                  vel_y, wall_norm_dict, cutoff=160) -> None:
#         super().__init__()
#         self._preprocess(wp, cutoff, 'wp')
#         self._preprocess(ws_u, cutoff,'ws_u')
#         self._preprocess(ws_w, cutoff,'ws_w')
#         self._preprocess(vel_y, cutoff, 'out')
#         self.inp = np.concatenate((self.wp, self.ws_u, self.ws_w), axis=1)
#         self.wall_norm_dict = wall_norm_dict
        
#     def _preprocess(self, data, cutoff, name):
#         data = (data[:, :, :cutoff]) if cutoff else data
#         setattr(self, name, (data).astype(np.float32))
               
#     def __len__(self):  
#         return self.out.shape[1]*self.out.shape[0]
    
#     def __getitem__(self, index):
#         ind = index % len(self.wall_norm_dict)
#         batch = index // len(self.wall_norm_dict)
#         return self.wall_norm_dict[ind], self.inp[batch], self.out[batch, ind:ind+1]

# class WMVF(Dataset):
#     def __init__(self, data, wall_norm_dict, cutoff=160) -> None:
#         super().__init__()
#         self._preprocess(data, cutoff, 'data')
#         self.wall_norm_dict = wall_norm_dict
        
#     def _preprocess(self, data, cutoff, name):
#         data = (data[:, :, :cutoff]) if cutoff else data
#         setattr(self, name, (data).astype(np.float32))
               
#     def __len__(self):  
#         return self.data.shape[0]
    
#     def __getitem__(self, index):
#         ind = index % len(self.wall_norm_dict)
#         batch = index // len(self.wall_norm_dict)
#         return self.wall_norm_dict[ind], self.data[batch, :3], self.data[batch, 3:]
    
# class WMVF_M(Dataset):
#     def __init__(self, data, cutoff=160) -> None:
#         super().__init__()
#         self._preprocess(data, cutoff, 'data')
        
#     def _preprocess(self, data, cutoff, name):
#         data = (data[:, :, :cutoff]) if cutoff else data
#         setattr(self, name, (data).astype(np.float32))
               
#     def __len__(self):  
#         return self.data.shape[0]
    
#     def __getitem__(self, index):
#         return self.data[index, :3], self.data[index, 3:]
    
class WMVF_P(Dataset):
    def __init__(self, data, patch_dims,
                 cutoff=160) -> None:
        super().__init__()
        self._preprocess(data, cutoff,'data')
        self.patch_dims = patch_dims
        self.space_ress = self.data.shape[-2:]
        
    def _preprocess(self, data, cutoff, name):
        data = (data[:, :, : ,:cutoff]) if cutoff else data
        setattr(self, name, (data).astype(np.float32))
               
    def __len__(self):  
        return self.data.shape[0]
    
    def __getitem__(self, index):
        start_indices = [np.random.randint(0,space_res-patch_dim) if space_res > patch_dim else 0 for space_res, patch_dim in zip(self.space_ress, self.patch_dims)]
        patch_x, patch_y = (slice(start_index, start_index+patch_dim) for start_index, patch_dim in zip(start_indices,self.patch_dims))
        return self.data[index, 1, ..., patch_x, patch_y], self.data[index, 0, ..., patch_x, patch_y]
    
class WMVF_baseline(Dataset):
    def __init__(self, data, wm_vf=True) -> None:
        super().__init__()
        self._preprocess(data, 'data')
        self.wm_vf = wm_vf
        
    def _preprocess(self, data, name):
        setattr(self, name, (data).astype(np.float32))
               
    def __len__(self):  
        return self.data.shape[0]
    
    def __getitem__(self, index):
        if self.wm_vf:
            return self.data[index, 0], self.data[index, 1] 
        else:
            return self.data[index, 1], self.data[index, 0]
    
# class WSVF(Dataset):
#     def __init__(self, ws_u, vel_y,
#                  wall_norm_dict, cutoff=160) -> None:
#         super().__init__()
#         self._preprocess(ws_u, cutoff,'ws_u')
#         self._preprocess(vel_y, cutoff, 'out')
#         self.inp = self.ws_u
#         self.wall_norm_dict = wall_norm_dict
        
#     def _preprocess(self, data, cutoff, name):
#         data = (data[:, :, :cutoff]) if cutoff else data
#         setattr(self, name, (data).astype(np.float32))
               
#     def __len__(self):  
#         return self.out.shape[1]*self.out.shape[0]
    
#     def __getitem__(self, index):
#         ind = index % len(self.wall_norm_dict)
#         batch = index // len(self.wall_norm_dict)
#         return self.wall_norm_dict[ind], self.inp[batch], self.out[batch, ind:ind+1]


# class VFVF(Dataset):
#     def __init__(self, vel_inp, vel_out,
#                  cutoff=160) -> None:
#         super().__init__()
#         self._preprocess(vel_inp, cutoff,'inp')
#         self._preprocess(vel_out, cutoff, 'out')
        
#     def _preprocess(self, data, cutoff, name):
#         data = (data[:, :, :cutoff]) if cutoff else data
#         setattr(self, name, (data).astype(np.float32))
               
#     def __len__(self):  
#         return self.out.shape[0]
    
#     def __getitem__(self, index):
#         return self.inp[index], self.out[index]

class VFVF(Dataset):
    def __init__(self, data, wall_norm_dict,
                 cutoff=160) -> None:
        super().__init__()
        self._preprocess(data, cutoff,'data')
        self.wall_norm_dict = wall_norm_dict
        
    def _preprocess(self, data, cutoff, name):
        data = (data[:, :, : ,:cutoff]) if cutoff else data
        setattr(self, name, (data).astype(np.float32))
               
    def __len__(self):  
        return self.data.shape[0]*(len(self.wall_norm_dict))
    
    def __getitem__(self, index):
        ind = index % len(self.wall_norm_dict)
        batch = index // len(self.wall_norm_dict)
        return np.array(self.wall_norm_dict[ind], dtype=np.float32), self.data[batch, ind], self.data[batch, ind+1]
    
class VFVF_patchify(Dataset):
    def __init__(self, data, wall_norm_dict, patch_dims,
                 cutoff=None) -> None:
        super().__init__()
        self._preprocess(data, cutoff,'data')
        self.wall_norm_dict = wall_norm_dict
        self.patch_dims = patch_dims
        self.space_ress = self.data.shape[-2:]
        
    def _preprocess(self, data, cutoff, name):
        data = (data[:, :, : ,:cutoff]) if cutoff else data
        setattr(self, name, (data).astype(np.float32))
               
    def __len__(self):  
        return self.data.shape[0]*(len(self.wall_norm_dict))
    
    def __getitem__(self, index):
        start_indices = [np.random.randint(0,space_res-patch_dim) if space_res > patch_dim else 0 for space_res, patch_dim in zip(self.space_ress, self.patch_dims)]
        ind = index % len(self.wall_norm_dict)
        batch = index // len(self.wall_norm_dict)
        patch_x, patch_y = (slice(start_index, start_index+patch_dim) for start_index, patch_dim in zip(start_indices,self.patch_dims))
        return np.array(self.wall_norm_dict[ind], dtype=np.float32), self.data[batch, ind, ..., patch_x, patch_y], self.data[batch, ind+1, ..., patch_x, patch_y]
    
class VFVF_patchify_2(Dataset):
    def __init__(self, data, wall_norm_dict, patch_dims,
                 cutoff=None) -> None:
        super().__init__()
        self._preprocess(data, cutoff,'data')
        self.wall_norm_dict = wall_norm_dict
        self.patch_dims = patch_dims
        self.space_ress = self.data.shape[-2:]
        self.time_steps = self.data.shape[0]
        self.ind_cumprod = np.concatenate((((np.cumprod((self.data.shape[:1] + (len(self.wall_norm_dict),) + self.data.shape[3:])[::-1]))[::-1])[1:], np.ones(1))).astype(int)
        assert len(self.patch_dims) == len(self.space_ress), "The patch dims list does not match the space res list"
        assert sum([self.patch_dims[i] > self.space_ress[i] for i in range(len(self.patch_dims))]) == 0, "the patch dims should be lower than space res!"
        
    def _preprocess(self, data, cutoff, name):
        data = (data[:, :, : ,:cutoff]) if cutoff else data
        setattr(self, name, (data).astype(np.float32))
               
    def __len__(self):  
        return self.time_steps * len(self.wall_norm_dict) * np.prod(self.space_ress) 
    
    def __getitem__(self, index):
        time_ind, y_ind, h_ind, w_ind = index // self.ind_cumprod[0], int((index // self.ind_cumprod[1])), index // self.ind_cumprod[2], index // self.ind_cumprod[3] 
        
        time_ind %=  self.time_steps
        y_ind %= len(self.wall_norm_dict)
        
        x0, x1 = self.data[time_ind, y_ind], self.data[time_ind, y_ind+1]
        wall_cond = np.array(self.wall_norm_dict[y_ind], dtype=np.float32)
        
        start_indices = []
        
        for space_res, probe_index in zip(self.space_ress, [h_ind, w_ind]):
            start_indices.append(probe_index % space_res)
            
        patches = [range(start_index, start_index + patch_dim) for start_index, patch_dim in zip(start_indices, self.patch_dims)]
        
        for i, patch in zip(range(-len(self.patch_dims),0), patches):
            x0, x1 = x0.take(indices=patch, axis=i, mode='wrap'), x1.take(indices=patch, axis=i, mode='wrap')
        
        return wall_cond, x0, x1
    
class WMAR(Dataset):
    def __init__(self, data,
                 cutoff=160) -> None:
        super().__init__()
        self._preprocess(data, cutoff,'data')
        
    def _preprocess(self, data, cutoff, name):
        data = data[:, :, :cutoff] if cutoff else data
        setattr(self, name, (data).astype(np.float32))
               
    def __len__(self):  
        return self.data.shape[0]*(self.data.shape[1] - 1)
    
    def __getitem__(self, index):
        pair_ind = index % (self.data.shape[1] - 1)
        time_ind = index // (self.data.shape[1] - 1)
        return self.data[time_ind, pair_ind:pair_ind+1], self.data[time_ind, pair_ind+1:pair_ind+2]
    
class WMAR_rollout(Dataset):
    def __init__(self, data,
                 rolling_steps,
                 cutoff=160) -> None:
        super().__init__()
        self.rolling_steps = rolling_steps
        self._preprocess(data, cutoff,'data')
        
    def _preprocess(self, data, cutoff, name):
        data = data[:, :, :cutoff] if cutoff else data
        setattr(self, name, (data).astype(np.float32))
               
    def __len__(self):
        assert self.rolling_steps < self.data.shape[1], "This rollout is not possible!"
        return self.data.shape[0]*(self.data.shape[1] - self.rolling_steps)
    
    def __getitem__(self, index):
        pair_ind = index % (self.data.shape[1] - self.rolling_steps)
        time_ind = index // (self.data.shape[1] - self.rolling_steps)
        return self.data[time_ind, pair_ind:pair_ind+1], self.data[time_ind, pair_ind+1:pair_ind+self.rolling_steps+1]
    
class Patched_Dataset(Dataset):
    def __init__(self, data, all_vel, patch_dims, multi_patch=False, zero_pad=True, *args) -> None:
        super().__init__()
        self.zero_pad = zero_pad
        if multi_patch:
            print("Using Multi-patch")
            self.sample_patch_dims = patch_dims[-1]
        self.multi_patch = multi_patch
        self.patch_dims = patch_dims
        self.space_ress = data.shape[-2:]
        self.time_steps = data.shape[0]
        # self.ind_cumprod = np.concatenate((((np.cumprod((self.data.shape[:1] + self.data.shape[2:])[::-1]))[::-1])[1:], np.ones(1))).astype(int)
        if not multi_patch:
            assert len(self.patch_dims) == len(self.space_ress), "The patch dims list does not match the space res list"
            assert sum([self.patch_dims[i] > self.space_ress[i] for i in range(len(self.patch_dims))]) == 0, "the patch dims should be lower than space res!"
        self._preprocess(data,'data')

    def _preprocess(self, data, name):#_preprocess(self, data, cutoff, name):
        if self.multi_patch:
            patch0, patch1 = self.patch_dims[-1][0], self.patch_dims[-1][1]
        else:
            patch0, patch1 = self.patch_dims[0], self.patch_dims[1]
        k0, k1 =  floor(self.space_ress[0]/patch0),  floor(self.space_ress[1]/patch1)
        pad0, pad1 = (k0+1)*patch0 - self.space_ress[0], (k1+1)*patch1 - self.space_ress[1],
        data = np.pad(data, ((0,0),(0,0),(pad0,pad0),(pad1,pad1)), mode='constant', constant_values=0.) if self.zero_pad else np.pad(data, ((0,0),(0,0),(pad0,pad0),(pad1,pad1)), mode='wrap')
        # data = (data[:, : ,:cutoff]) if cutoff else data
        x_coord, z_coord = np.linspace(-1, 1, self.space_ress[0]+2*pad0), np.linspace(-1, 1, self.space_ress[1]+2*pad1)
        x_coord, z_coord = x_coord[None, :, None] * np.ones((1, self.space_ress[0]+2*pad0, self.space_ress[1]+2*pad1)),\
                           z_coord[None, None, :] * np.ones((1, self.space_ress[0]+2*pad0, self.space_ress[1]+2*pad1))
        setattr(self, name, (data).astype(np.float32))
        setattr(self, "x_coord", (x_coord).astype(np.float32))
        setattr(self, "z_coord", (z_coord).astype(np.float32))
        setattr(self, "pads", (pad0, pad1) )
        
    def __len__(self):  
        return self.time_steps #* np.prod(self.space_ress) 
    
    def __getitem__(self, index):
        patch_dims = self.patch_dims if not hasattr(self, "sample_patch_dims") else self.sample_patch_dims
        time_ind, h_ind, w_ind = index, np.random.randint(0, self.space_ress[0]+ 2*self.pads[0] - patch_dims[0]), np.random.randint(0, self.space_ress[1]+ 2*self.pads[1] - patch_dims[1]) 
        #index // self.ind_cumprod[1], index // self.ind_cumprod[2] 
        
        time_ind %=  self.time_steps
        
        x1 = self.data[time_ind]
        
        start_indices = [h_ind, w_ind]
        # for space_res, probe_index in zip(self.space_ress, [h_ind, w_ind]):
        #     start_indices.append(probe_index % space_res)
            
        patches = [range(start_index, start_index + patch_dim) for start_index, patch_dim in zip(start_indices, patch_dims)]
        
        x_coord, z_coord = self.x_coord.copy(), self.z_coord.copy()
        for i, patch in zip(range(-len(patch_dims),0), patches):
            x1 = x1.take(indices=patch, axis=i, mode='wrap')
            x_coord = x_coord.take(indices=patch, axis=i, mode='wrap')
            z_coord = z_coord.take(indices=patch, axis=i, mode='wrap')
        
        x1 = np.concat([x1, x_coord, z_coord], axis=0)
        
        return np.empty_like(x1), x1
    
class Patched_Dataset_W(Dataset):
    def __init__(self, data, patch_dims, multi_patch=False, zero_pad=True, multiple_of_time_steps=1, *args, **kwargs) -> None:
        super().__init__()
        self.zero_pad = zero_pad
        if multi_patch:
            print("Using Multi-patch")
            self.sample_patch_dims = patch_dims[-1]
        self.multi_patch = multi_patch
        self.patch_dims = patch_dims
        self.space_ress = data.shape[:2]
        self.time_steps = data.shape[-1]
        self.mutiple_of_time_steps = multiple_of_time_steps
        if "re" in kwargs.keys() and kwargs["re"] is not None:
            self.re = kwargs["re"]
            self.inner_length_scale = {180:0.18053097561027237, 500:0.4920473773265651, 1000:1}
        else:
            self.re = None
        # self.ind_cumprod = np.concatenate((((np.cumprod((self.data.shape[:1] + self.data.shape[2:])[::-1]))[::-1])[1:], np.ones(1))).astype(int)
        if not multi_patch:
            assert len(self.patch_dims) == len(self.space_ress), "The patch dims list does not match the space res list"
            assert sum([self.patch_dims[i] > self.space_ress[i] for i in range(len(self.patch_dims))]) == 0, "the patch dims should be lower than space res!"
        self._preprocess(data,'data')

    def _preprocess(self, data, name):#_preprocess(self, data, cutoff, name):
        if self.multi_patch:
            patch0, patch1 = self.patch_dims[-1][0], self.patch_dims[-1][1]
        else:
            patch0, patch1 = self.patch_dims[0], self.patch_dims[1]
        k0, k1 =  floor(self.space_ress[0]/patch0),  floor(self.space_ress[1]/patch1)
        pad0, pad1 = (k0+1)*patch0 - self.space_ress[0], (k1+1)*patch1 - self.space_ress[1],
        data = np.pad(data, ((pad0,pad0),(pad1,pad1),(0,0)), mode='constant', constant_values=0.) if self.zero_pad else np.pad(data, ((pad0,pad0),(pad1,pad1),(0,0)), mode='wrap')
        x_coord, z_coord = np.linspace(-1, 1, self.space_ress[0]+2*pad0), np.linspace(-1, 1, self.space_ress[1]+2*pad1)
        if self.re is not None:
            x_coord, z_coord = self.inner_length_scale[self.re]*x_coord, self.inner_length_scale[self.re]*z_coord
        x_coord, z_coord = x_coord[:, None] * np.ones((self.space_ress[0]+2*pad0, self.space_ress[1]+2*pad1)),\
                           z_coord[None, :] * np.ones((self.space_ress[0]+2*pad0, self.space_ress[1]+2*pad1))
        setattr(self, name, (data).astype(np.float32))
        setattr(self, "x_coord", (x_coord).astype(np.float32))
        setattr(self, "z_coord", (z_coord).astype(np.float32))
        setattr(self, "pads", (pad0, pad1) )
        
    def __len__(self):  
        return self.mutiple_of_time_steps * self.time_steps #* np.prod(self.space_ress) 
    
    def __getitem__(self, index):
        patch_dims = self.patch_dims if not hasattr(self, "sample_patch_dims") else self.sample_patch_dims
        time_ind, h_ind, w_ind = index % self.time_steps, np.random.randint(0, self.space_ress[0]+ 2*self.pads[0] - patch_dims[0]), np.random.randint(0, self.space_ress[1]+ 2*self.pads[1] - patch_dims[1])        
        time_ind %=  self.time_steps
        
        x1 = self.data[..., time_ind]
        
        start_indices = [h_ind, w_ind]
            
        patches = [range(start_index, start_index + patch_dim) for start_index, patch_dim in zip(start_indices, patch_dims)]
        
        x_coord, z_coord = self.x_coord.copy(), self.z_coord.copy()
        for i, patch in zip(range(0, len(patch_dims)), patches):
            x1 = x1.take(indices=patch, axis=i, mode='wrap')
            x_coord = x_coord.take(indices=patch, axis=i, mode='wrap')
            z_coord = z_coord.take(indices=patch, axis=i, mode='wrap')
        
        x1 = np.stack([x1, x_coord, z_coord], axis=0)
        
        return np.empty_like(x1), x1
    
class Dataset_W(Dataset):
    def __init__(self, data, patch_dims, *args) -> None:
        super().__init__()
        self._preprocess(data,'data')
        self.time_steps = data.shape[-1]

    def _preprocess(self, data, name):#_preprocess(self, data, cutoff, name):
        setattr(self, name, (data).astype(np.float32))
        
    def __len__(self):  
        return self.time_steps
    
    def __getitem__(self, index):
        
        x1 = self.data[..., index]
        x1 = x1[None]
        return np.empty_like(x1), x1
    
class Stitched_Dataset_1(Dataset):
    
    def __init__(self, vf_paths, N):
        super().__init__()
        self.vf_paths = vf_paths
        self.N = N
        self.std_list = [None]*len(vf_paths)
        
    def load_data(self, y_index): # load only the velocity planes in consideration in memory 
        assert y_index != 0 and y_index != len(self.vf_paths) - 1, "can only take intermediate indices" 
        data = [] 
        for i in range(-1, 2):
            intr = np.load((self.vf_paths[y_index + i]))
            if self.std_list[y_index + i] is not None:
                data.append(intr/self.std_list[y_index + i])
            else:
                std = intr.std()
                self.std_list[y_index + i] = std
                data.append(intr/std)
        data = np.concat(data, axis=1)
        setattr(self, "data", data.astype(np.float32))
        
    def __len__(self):
        return  self.N
    
    def __getitem__(self, index):
        
        x1 = self.data[index]
        
        return np.empty_like(x1), x1
    
class Stitched_Dataset_2(Dataset):
    
    def __init__(self, vf_paths, N, y_delta_list=None):
        super().__init__()
        self.vf_paths = vf_paths
        self.N = N
        self._load_all_data()
        if y_delta_list is None:
            self.y_delta_list = None
        else:
            self.y_delta_list = np.array(y_delta_list).astype(np.float32)
        
    def _load_all_data(self):
        data = np.concat([np.load(f) for f in self.vf_paths], axis=1)
        data = data/data.std(axis=(0,2,3), keepdims=True)
        setattr(self, "data", data.astype(np.float32))
        
    def __len__(self):
        return  self.N
    
    def __getitem__(self, index):
        
        x1 = self.data[index]
        
        if self.y_delta_list is None:
            return np.empty_like(x1), x1
        else:
            return np.empty_like(x1), x1, self.y_delta_list


class DarcyFlow(Dataset):
    """
    Dataset for Darcy Flow HDF5 files.

    Darcy flow describes fluid flow through porous media, governed by:
        -∇·(ν(x)∇u(x)) = f(x)

    where ν(x) is the spatially-varying permeability/viscosity coefficient,
    and u(x) is the solution (pressure/flow field).

    HDF5 file structure:
        - 'nu': Input field (permeability), shape (N, H, W)
        - 'tensor': Output field (solution), shape (N, 1, H, W) or (N, H, W)
        - 'x-coordinate': Optional x-coordinates
        - 'y-coordinate': Optional y-coordinates

    Args:
        hdf5_path: Path to HDF5 file
        input_key: Key for input data in HDF5 (default: 'nu')
        output_key: Key for output data in HDF5 (default: 'tensor')
        contrastive: Enable contrastive learning mode
        normalize: Apply standardization (mean=0, std=1)
        use_eqm_format: If True, return (empty, x1) for EQM training
                        If False, return (x0, x1) as input-output pair
    """
    def __init__(self, hdf5_path, input_key='nu', output_key='tensor',
                 contrastive=False, normalize=True, use_eqm_format=True):
        super().__init__()
        import h5py

        self.contrastive = contrastive
        self.use_eqm_format = use_eqm_format

        # Load data from HDF5
        with h5py.File(hdf5_path, 'r') as f:
            input_data = np.array(f[input_key]).astype(np.float32)
            output_data = np.array(f[output_key]).astype(np.float32)

        # Ensure input has channel dimension: (N, H, W) -> (N, 1, H, W)
        if len(input_data.shape) == 3:
            input_data = input_data[:, np.newaxis, :, :]

        # Ensure output has channel dimension: (N, H, W) -> (N, 1, H, W)
        if len(output_data.shape) == 3:
            output_data = output_data[:, np.newaxis, :, :]

        # Store original data
        self.input_data = input_data
        self.output_data = output_data

        # Normalize if requested
        if normalize:
            # Normalize input per-channel across spatial dimensions
            input_mean = input_data.mean(axis=(0, 2, 3), keepdims=True)
            input_std = input_data.std(axis=(0, 2, 3), keepdims=True)
            self.input_data = (input_data - input_mean) / (input_std + 1e-8)

            # Normalize output per-channel across spatial dimensions
            output_mean = output_data.mean(axis=(0, 2, 3), keepdims=True)
            output_std = output_data.std(axis=(0, 2, 3), keepdims=True)
            self.output_data = (output_data - output_mean) / (output_std + 1e-8)

        self.n = len(self.input_data)
        self.shape = self.output_data.shape[1:]  # (C, H, W)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        if self.use_eqm_format:
            # For EQM: x0 is empty (base distribution), x1 is target
            # We'll use the output (solution) as x1
            x0 = np.empty(self.shape, dtype=np.float32)
            x1 = self.output_data[index]
        else:
            # For conditional flow matching: x0 is input, x1 is output
            x0 = self.input_data[index]
            x1 = self.output_data[index]

        if self.contrastive:
            random_index = np.random.randint(0, self.n)
            while index == random_index:
                random_index = np.random.randint(0, self.n)

            if self.use_eqm_format:
                x0_cont = np.empty(self.shape, dtype=np.float32)
                x1_cont = self.output_data[random_index]
            else:
                x0_cont = self.input_data[random_index]
                x1_cont = self.output_data[random_index]

            return x0, x1, x0_cont, x1_cont

        return x0, x1


DATASETS = {"WP":None,"KS":None, "WPWS":None, "WPWS_DD":None,
            "Joint" : Joint,
            "VF_FM":VF_FM,
            "VF_FM_patchify":VF_FM_patchify,
            "WMVF":None, "WMVF_M": None, "WMVF_P": WMVF_P,
            "WMVF_baseline":WMVF_baseline,
            "WSVF":None,
            "VFVF":VFVF, "VFVF_P":VFVF_patchify,
            "VFVF_P2": VFVF_patchify_2,
            "WMAR":WMAR, "WMARR":WMAR_rollout, "Patched":Patched_Dataset,
            "Patched_W":Patched_Dataset_W, "Baseline_W":Dataset_W, "Stitched": Stitched_Dataset_2,
            "DarcyFlow": DarcyFlow}


# class WPWS_Sensor(Dataset):
#     def __init__(self, wp, ws, mean_wp, std_wp, mean_ws, std_ws, cutoff=160, sensor1_count=2, sensor2_count=2) -> None:
#         super().__init__()
#         self._preprocess(wp, cutoff, 'wp')
#         self._norm(mean_wp, std_wp, 'wp')
#         self._preprocess(ws, cutoff,'ws')
#         self._norm(mean_ws, std_ws, 'ws')       
#         self.mask_shape = (self.wp[0]).shape 
#         self.sensor1_count = sensor1_count
#         self.sensor2_count = sensor2_count

#     def _preprocess(self, data, cutoff, name):
#         data = rearrange(data, "b l c w -> (b c) l w")
#         data = (data[:, :cutoff])[:, None] if cutoff else data[:, None]
#         setattr(self, name, (data).astype(np.float32))
        
#     def _norm(self, mean, std, name):
#         data = getattr(self, name)
#         data = (data - mean)/std
#         setattr(self, name, data)
               
#     def __len__(self):  
#         return self.wp.shape[0]
    
#     def __getitem__(self, index):
#         mask_x, mask_y = np.random.randint(0, self.mask_shape[1], self.sensor1_count), np.random.randint(0, self.mask_shape[2], self.sensor2_count)
#         mask = np.zeros_like(self.mask_shape)
#         mask[mask_x, mask_y] = 1.
#         return self.wp[index]*mask.astype(np.float32), self.ws[index]

if __name__ == "__main__":
    a = VFVF_patchify(np.random.randn(100, 2, 1, 20, 20), {0:0}, patch_dims=[20, 20], cutoff=None)
    a[19]