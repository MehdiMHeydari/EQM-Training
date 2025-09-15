import numpy as np
from torch.utils.data import DataLoader

def quickload(fp, csize):
    
    def load_npy_in_chunks(filepath, chunk_size):
        """Load a .npy file in chunks using memory mapping."""

        with open(filepath, 'rb') as f:
            version = np.lib.format.read_magic(f)
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)

        # Calculate number of chunks
            num_chunks = np.ceil(shape[0] / chunk_size).astype(int)

            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, shape[0])

                # Use memory mapping to load the chunk
                chunk = np.memmap(filepath, dtype=dtype, mode='r', shape=(end - start,) + shape[1:], offset=f.tell())

                yield chunk
    
    return np.concat([c for c in load_npy_in_chunks(fp, csize)])
            
# def get_loaders(datapath, batch_size, dataset_):
    
#     data = np.load(datapath)
#     mean, std = np.mean(data), np.std(data)
#     train_dataloader = DataLoader(dataset_(data, mean, std), batch_size=batch_size, shuffle=True)
    
#     return train_dataloader

# def get_loaders_wpws(wp_path, ws_path, batch_size, cutoff, uniform_prob, uniform_prob_list, dataset_):
    
#     wp = np.load(wp_path)
#     ws = np.load(ws_path)
#     mean_wp, std_wp = np.mean(wp), np.std(wp)
#     mean_ws, std_ws = np.mean(ws), np.std(ws)
#     train_dataloader = DataLoader(dataset_(wp, ws, mean_wp, std_wp, mean_ws, std_ws, cutoff, uniform_prob, uniform_prob_list), batch_size=batch_size, shuffle=True)
    
#     return train_dataloader

# def get_loaders_wmvf(wp_path, ws_u_path, ws_w_path, vf_path, batch_size, cutoff, wall_norm_dict, dataset_):
    
#     def norm(d, m, s):
#         return (d-m)/s
    
#     wp = np.load(wp_path)
#     ws_u = np.load(ws_u_path)[:, None]
#     ws_w = np.load(ws_w_path)[:, None]
#     vf = np.load(vf_path)
#     mean_wp, std_wp = np.mean(wp), np.std(wp)
#     mean_ws_u, std_ws_u = np.mean(ws_u), np.std(ws_u)
#     mean_ws_w, std_ws_w = np.mean(ws_w), np.std(ws_w)
#     mean_vf, std_vf = np.mean(vf, axis=(0,2,3), keepdims=True), np.std(vf, axis=(0,2,3), keepdims=True)
    

#     wp = norm(wp, mean_wp, std_wp)
#     ws_u = norm(ws_u, mean_ws_u, std_ws_u)
#     ws_w = norm(ws_w, mean_ws_w, std_ws_w)
#     vf = norm(vf, mean_vf, std_vf)

#     train_dataloader = DataLoader(dataset_(wp[:30000], ws_u[:30000], ws_w[:30000], vf[:30000], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)
#     test_dataloader = DataLoader(dataset_(wp[30000:], ws_u[30000:], ws_w[30000:], vf[30000:], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)

#     return train_dataloader, test_dataloader

def get_loaders_wmvf(wm_paths, vf_paths, batch_size, time_cutoff, cutoff, wall_norm_dict, dataset_, scale_inputs=False):
    
    def norm(d, m, s):
        return (d-m)/s
    
    data = []
    for path in wm_paths:
        d = np.load(path)
        data.append(d)
        
    for path in vf_paths:
        d = np.load(path)   
        data.append(d)
    
    data = np.concatenate(data, axis=1)
    m, s = np.mean(data, axis=(0,2,3), keepdims=True), np.std(data, axis=(0,2,3), keepdims=True)
    
    data = norm(data, m, s)

    train_dataloader = DataLoader(dataset_(data[:time_cutoff], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_(data[36000:39900], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def get_loaders_wmvf_multi(wm_paths, vf_paths, batch_size, time_cutoff, cutoff, dataset_):
    
    def norm(d, m, s):
        return (d-m)/s
    
    data = []
    for path in wm_paths:
        d = np.load(path)
        data.append(d)
        
    for path in vf_paths:
        d = np.load(path)   
        data.append(d)
    
    data = np.concatenate(data, axis=1)
    m, s = np.mean(data, axis=(0,2,3), keepdims=True), np.std(data, axis=(0,2,3), keepdims=True)
    
    data = norm(data, m, s)

    train_dataloader = DataLoader(dataset_(data[:time_cutoff], cutoff), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_(data[time_cutoff:], cutoff), batch_size=batch_size, shuffle=True) #DataLoader(dataset_(data[36000:39900], cutoff), batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def get_loaders_wmvf_patch(wm_paths, vf_paths, batch_size, time_cutoff, cutoff, patch_dims, dataset_, jump=1, scale_inputs=False):
    
    def norm(d, m, s):
        if not scale_inputs:
            return (d-m)/s
        else: # only scale the outputs
            d[:, 0] = (d[:, 0] - m[:, 0])/s[:, 0]
            return d
        
    wm_data = []
    for pxy_path in wm_paths:
        pxy_data = []    
        for path in pxy_path:
            d = np.load(path)
            pxy_data.append(d)   
        wm_data.append(pxy_data)
    if len(wm_data) == 1:    
        wm_data = np.concat(wm_data[0], axis=1)[:, None]
    else:
        wm_data = [np.concat(wm, axis=1) for wm in wm_data]
        wm_data = np.stack(wm_data, axis=1)
    
    vf_data = []  
    for uvw_path in vf_paths:
        uvw_data = []    
        for path in uvw_path:
            d = np.load(path)
            uvw_data.append(d)   
        vf_data.append(uvw_data)
    if len(vf_data) == 1:
        vf_data = np.concat(vf_data[0], axis=1)[:, None]
    else:
        vf_data = [np.concat(vf, axis=1) for vf in vf_data]
        vf_data = np.stack(vf_data, axis=1)
    
    data = np.concatenate([wm_data, vf_data], axis=1)
    m, s = np.mean(data, axis=(0,3,4), keepdims=True), np.std(data, axis=(0,3,4), keepdims=True)
    # scaling v and w velocity components with u velocity component (Guastoni et al. 2021)
    if scale_inputs:
        for i in range(1,3):
            data[:, 1, i] = data[:, 1, i] * s[:, 1, 0]/s[:, 1, i]
            s[:, 1, i] = s[:, 1, 0]
    data = norm(data, m, s)

    train_dataloader = DataLoader(dataset_(data[:time_cutoff:jump], patch_dims, cutoff), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_(data[-1000:], patch_dims, cutoff), batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def get_loaders_wmvf_baseline(wm_paths, vf_paths, batch_size, time_cutoff, dataset_, jump=1, scale_inputs=False, wm_vf=True):
    
    def norm(d, m, s):
        if not scale_inputs:
            return (d-m)/s
        else: # only scale the outputs
            d[:, 0] = (d[:, 0] - m[:, 0])/s[:, 0]
            return d
        
    wm_data = []
    for pxy_path in wm_paths:
        pxy_data = []    
        for path in pxy_path:
            d = np.load(path)
            pxy_data.append(d)   
        wm_data.append(pxy_data)
    if len(wm_data) == 1:    
        wm_data = np.concat(wm_data[0], axis=1)[:, None]
    else:
        wm_data = [np.concat(wm, axis=1) for wm in wm_data]
        wm_data = np.stack(wm_data, axis=1)
    
    vf_data = []  
    for uvw_path in vf_paths:
        uvw_data = []    
        for path in uvw_path:
            d = np.load(path)
            uvw_data.append(d)   
        vf_data.append(uvw_data)
    if len(vf_data) == 1:
        vf_data = np.concat(vf_data[0], axis=1)[:, None]
    else:
        vf_data = [np.concat(vf, axis=1) for vf in vf_data]
        vf_data = np.stack(vf_data, axis=1)
    
    data = np.concatenate([wm_data, vf_data], axis=1)
    m, s = np.mean(data, axis=(0,3,4), keepdims=True), np.std(data, axis=(0,3,4), keepdims=True)
    # scaling v and w velocity components with u velocity component (Guastoni et al. 2021)
    if scale_inputs:
        for i in range(1,3):
            data[:, 1, i] = data[:, 1, i] * s[:, 1, 0]/s[:, 1, i]
            s[:, 1, i] = s[:, 1, 0]
    data = norm(data, m, s)

    train_dataloader = DataLoader(dataset_(data[:time_cutoff:jump], wm_vf), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_(data[-1000:], wm_vf), batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

# def get_loaders_wsvf(ws_u_path, vf_path, batch_size, cutoff, wall_norm_dict, dataset_):
    
#     def norm(d, m, s):
#         return (d-m)/s
    
#     ws_u = np.load(ws_u_path)[:, None]
#     vf = np.load(vf_path)
#     mean_ws_u, std_ws_u = np.mean(ws_u), np.std(ws_u)
#     mean_vf, std_vf = np.mean(vf, axis=(0,2,3), keepdims=True), np.std(vf, axis=(0,2,3), keepdims=True)
    

#     ws_u = norm(ws_u, mean_ws_u, std_ws_u)
#     vf = norm(vf, mean_vf, std_vf)

#     # train_dataloader = DataLoader(dataset_(ws_u[:30000], vf[:30000], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)
#     # test_dataloader = DataLoader(dataset_(ws_u[30000:], vf[30000:], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)
    
#     train_dataloader = DataLoader(dataset_(ws_u, vf, wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)
#     test_dataloader = DataLoader(dataset_(ws_u[35000:], vf[35000:], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)

#     return train_dataloader, test_dataloader

# def get_loaders_vfvf(u_inpp, u_outp,
#                      v_inpp, v_outp,
#                      w_inpp, w_outp,
#                      batch_size, cutoff, dataset_):
    
#     def norm(d, m, s):
#         return (d-m)/s
    
#     u_inp = np.load(u_inpp)
#     u_out = np.load(u_outp)
    
#     v_inp = np.load(v_inpp)
#     v_out = np.load(v_outp)
    
#     w_inp = np.load(w_inpp)
#     w_out = np.load(w_outp)
    
#     inp = np.concatenate((u_inp, v_inp, w_inp),axis=1)
#     out = np.concatenate((u_out, v_out, w_out),axis=1)

#     mean_inp, std_inp = np.mean(inp, axis=(0,2,3), keepdims=True), np.std(inp, axis=(0,2,3), keepdims=True)
#     mean_out, std_out = np.mean(out, axis=(0,2,3), keepdims=True), np.std(out, axis=(0,2,3), keepdims=True)


#     inp = norm(inp, mean_inp, std_inp)
#     out = norm(out, mean_out, std_out)

#     train_dataloader = DataLoader(dataset_(inp, out, cutoff), batch_size=batch_size, shuffle=True)

#     return train_dataloader

# def get_loaders_vfvf(vf_paths, batch_size, time_cutoff, cutoff, wall_norm_dict, dataset_):
    
#     def norm(d, m, s):
#         return (d-m)/s

#     data = []
#     for uvw_path in vf_paths:
#         uvw_data = []    
#         for path in uvw_path:
#             d = np.load(path)
#             uvw_data.append(d)   
#         data.append(uvw_data)
#     if len(data[0]) > 1:
#         data = [np.concatenate(uvw, axis=1) for uvw in data]
#         data = np.stack(data, axis=1)
#         m, s = np.mean(data, axis=(0,3,4), keepdims=True), np.std(data, axis=(0,3,4), keepdims=True)
#     else:
#         data = np.concatenate(data,axis=2)
#         data = data[0]
#         m, s = np.mean(data, axis=(0,2,3), keepdims=True), np.std(data, axis=(0,2,3), keepdims=True)
    
#     data = norm(data, m, s)

#     if len(data.shape) == 4:
#         data = data[:, :, None]
#     train_dataloader = DataLoader(dataset_(data[:time_cutoff], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)
#     test_dataloader = DataLoader(dataset_(data[time_cutoff:], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True) #DataLoader(dataset_(data[36000:39900], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)
       
    
#     return train_dataloader, test_dataloader

def get_joint_loaders(vf_paths, batch_size, dataset_, contrastive=False):
    
    def norm(d, m, s):
        return (d-m)/s
    
    data = []
    
    for uvw_path in vf_paths:
        uvw_data = []    
        for path in uvw_path:
            d = np.load(path)
            uvw_data.append(d)   
        data.append(uvw_data)
    data = [np.concatenate(uvw, axis=1) for uvw in data]
    
    data = np.concatenate(data, axis=1)
    m, s = np.mean(data, axis=(0,2,3), keepdims=True), np.std(data, axis=(0,2,3), keepdims=True)

    data = norm(data, m, s)
    train_dataloader = DataLoader(dataset_(data, contrastive=contrastive), batch_size=batch_size, shuffle=True)
       
    return train_dataloader

def get_stitching_loaders(vf_paths, data_size, batch_size, dataset_):
    
    train_dataloader = DataLoader(dataset_(vf_paths, data_size), batch_size=batch_size, shuffle=True)
       
    return train_dataloader

def get_loaders_vf_fm(vf_paths, batch_size, dataset_, jump=1, all_vel=True, spatial_cutoff=None, z_spatial_cutoff=None, time_cutoff=None, patch_dims=None,
                      multi_patch=False, zero_pad=True):
    
    def norm(d, m, s):
        return (d-m)/s

    data = []
    
    if (len(vf_paths) == 3 and type(vf_paths[0]) == str):
        for path in vf_paths:
            d = np.load(path)
            data.append(d)
        data = np.concatenate(data, axis=1)
        data = data[..., :z_spatial_cutoff] if z_spatial_cutoff is not None else data
        m, s = np.mean(data, axis=(0,2,3), keepdims=True), np.std(data, axis=(0,2,3), keepdims=True)
            
    elif (len(vf_paths) == 1 and type(vf_paths[0][0]) == str):
        vf_paths = vf_paths[0]
        for path in vf_paths:
            d = np.load(path)
            if d.ndim == 3:
                d = d[:, None]
            data.append(d)
        data = np.concatenate(data, axis=1)
        data = data[:, :, :spatial_cutoff] if spatial_cutoff is not None else data
        data = data[..., :z_spatial_cutoff] if z_spatial_cutoff is not None else data
        m, s = np.mean(data, axis=(0,2,3), keepdims=True), np.std(data, axis=(0,2,3), keepdims=True)

    else:
        for uvw_path in vf_paths:
            uvw_data = []    
            for path in uvw_path:
                d = np.load(path)
                uvw_data.append(d)   
            data.append(uvw_data)
        data = [np.concatenate(uvw, axis=1) for uvw in data]
        data = np.stack(data, axis=1)
        data = data[:, :, :, :spatial_cutoff] if spatial_cutoff is not None else data
        data = data[..., :z_spatial_cutoff] if z_spatial_cutoff is not None else data
        m, s = np.mean(data, axis=(0,3,4), keepdims=True), np.std(data, axis=(0,3,4), keepdims=True)
    
    data = norm(data, m, s)

    if patch_dims is not None:
        if time_cutoff is not None:
            train_dataloader = DataLoader(dataset_(data[:time_cutoff:jump], all_vel, patch_dims, multi_patch, zero_pad), batch_size=batch_size, shuffle=True)
        else:
            train_dataloader = DataLoader(dataset_(data[::jump], all_vel, patch_dims, multi_patch, zero_pad), batch_size=batch_size, shuffle=True)
    else:
        train_dataloader = DataLoader(dataset_(data[:time_cutoff:jump], all_vel), batch_size=batch_size, shuffle=True)
       
    return train_dataloader

def get_loaders_wp_fm(wall_pres_path, nx, nz, nstep, batch_size, dataset_, patch_dims, multi_patch, zero_pad, spatial_start=0, spatial_cutoff=None,
                      temporal_cutoff=None, re=None, not_dat_file=False):
    
    inner_scaling = {180 : 0.0006499372706408232/1.5, 500: 0.0004886492934049004, 1000: 0.00041547712259805273}

    def norm(d, m, s):
        if re is not None:
            fluc = (d - m)
            fluc = fluc/inner_scaling[re]
            fluc = fluc/fluc.std()
            return fluc
        else:
            return (d-m)/s
    
    def read_data(file_path, nx, nz, nstep):
        data  = np.fromfile(file_path, dtype=np.float64, count=nx*nz*nstep).reshape((nx, nz, nstep), order='F')
        return data
    
    if not_dat_file:
        data = np.load(wall_pres_path) # shape as (nx, nz, nstep)
    else:
        data = read_data(wall_pres_path, nx, nz, nstep)
    
    data = data[spatial_start:spatial_cutoff] if spatial_cutoff is not None else data[spatial_start:-1, :-1]

    m, s = data.mean(), data.std()
    
    data = norm(data, m, s)

    if multi_patch is not None:
        if temporal_cutoff is not None:
                train_dataloader = DataLoader(dataset_(data[..., :temporal_cutoff], patch_dims, multi_patch, zero_pad), batch_size=batch_size, shuffle=True)      
        else:
            train_dataloader = DataLoader(dataset_(data, patch_dims, multi_patch, zero_pad), batch_size=batch_size, shuffle=True)
    else:
        train_dataloader = DataLoader(dataset_(data, patch_dims), batch_size=batch_size, shuffle=True)
       
    return train_dataloader

def get_loaders_vfvf(vf_paths, batch_size, time_cutoff, cutoff, wall_norm_dict, patch_dims, dataset_, jump=1):
    
    def norm(d, m, s):
        return (d-m)/s

    data = []
    for uvw_path in vf_paths:
        uvw_data = []    
        for path in uvw_path:
            d = np.load(path)
            uvw_data.append(d)   
        data.append(uvw_data)
    if len(data[0]) > 1:
        data = [np.concatenate(uvw, axis=1) for uvw in data]
        data = np.stack(data, axis=1)
        m, s = np.mean(data, axis=(0,3,4), keepdims=True), np.std(data, axis=(0,3,4), keepdims=True)
    else:
        data = np.concatenate(data,axis=2)
        data = data[0]
        m, s = np.mean(data, axis=(0,2,3), keepdims=True), np.std(data, axis=(0,2,3), keepdims=True)
    
    data = norm(data, m, s)

    if len(data.shape) == 4:
        data = data[:, :, None]
    train_dataloader = DataLoader(dataset_(data[:time_cutoff:jump], wall_norm_dict, patch_dims, cutoff), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_(data[-1000:], wall_norm_dict, patch_dims=[320,200], cutoff=None), batch_size=10, shuffle=True) #DataLoader(dataset_(data[36000:39900], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)
       
    return train_dataloader, test_dataloader

# def get_loaders_wmar(wm_paths, v_0_150_path, v_int_paths, batch_size, cutoff, dataset_):
    
#     def norm(d, m, s):
#         return (d-m)/s
    
#     v_0_150 = np.load(v_0_150_path)
    
#     data = []
#     for path in wm_paths:
#         d = np.load(path)
#         data.append(d[:, None] if len(d.shape) !=4 else d)
        
#     data.append(v_0_150[:, 0:1])
    
#     for path in v_int_paths:
#         d = np.load(path)
#         data.append(d[:, None] if len(d.shape) !=4 else d)
        
#     data.append(v_0_150[:, 1:])
    
#     data = np.concatenate(data, axis=1)
#     m, s = np.mean(data, axis=(0,2,3), keepdims=True), np.std(data, axis=(0,2,3), keepdims=True)

#     data = norm(data, m, s)
    
#     train_dataloader = DataLoader(dataset_(data, cutoff), batch_size=batch_size, shuffle=True)

#     return train_dataloader

def get_loaders_wmar(wm_paths, v_int_paths, rolling_steps, batch_size, cutoff, dataset_, time_cutoff):
    
    def norm(d, m, s):
        return (d-m)/s
        
    data = []
    for path in wm_paths:
        d = np.load(path)
        data.append(d[:time_cutoff, None] if len(d.shape) !=4 else d[:time_cutoff])
            
    for path in v_int_paths:
        d = np.load(path)
        data.append(d[:time_cutoff, None] if len(d.shape) !=4 else d[:time_cutoff])
            
    data = np.concatenate(data, axis=1)
    m, s = np.mean(data, axis=(0,2,3), keepdims=True), np.std(data, axis=(0,2,3), keepdims=True)

    data = norm(data, m, s)
    
    train_dataloader = DataLoader(dataset_(data, rolling_steps, cutoff), batch_size=batch_size, shuffle=True)

    return train_dataloader