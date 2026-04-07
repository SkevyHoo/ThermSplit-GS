
import os

import torch
import torch.nn as nn
import roma
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import CameraInfo
from scene.bezier import BezierModel
from arguments import ModelParams
import utils.pytorch3d_functions as torch3d
from scene.cameras import Camera, MiniCam
from utils.camera_utils import cameraList_from_camInfos
import gaussian_renderer
from scene.gaussian_activation import inverse_sigmoid
from scene.weight_gen import WeightGenerator

class ThermalCoupleModel(nn.Module):
    
    def __init__(self, cam_infos:list, args:ModelParams):
        super(ThermalCoupleModel, self).__init__()
        
        """
        Aliases:
        C : curve order
        n : # imgs.
        f : # subframes.
        """
        print("Initializing Curve Model...")
        
        self.USE_REF_CAM = args.USE_REF_CAM
        
        self.WG_FLAG = False
        
        C = self.curve_order = args.curve_order
        self.f = self.n_subframes = args.num_subframes
        self.curve_type = args.curve_type
        self.curve_random_sample = args.curve_random_sample
        self.gaussians = None

        self.original_cam = cameraList_from_camInfos(cam_infos, 
                                                     resolution_scale=1, 
                                                     args=args)
        
        self.use_weight_generator = args.use_weight_generator
        if self.use_weight_generator:
            self.weight_function = WeightGenerator(args.num_subframes, args.num_subframes).cuda()
                                            

        rotations = []
        translations = []
        
        for cam_info in cam_infos:
            cam_info: CameraInfo
            
            # cam_info.R/T are in world-to-cam transform.
            # we need to convert to cam-to-world transform.
            rotations.append(torch.from_numpy(cam_info.R)) # originally transposed to [n,3,3] matrix.
            translations.append(torch.from_numpy(-cam_info.T@cam_info.R.T)) # translation vec of c2w: cam location. [n,3]

        rotations = torch.stack(rotations).cuda()
        translations = torch.stack(translations).cuda() 

        # Initial Curve Parameters
        self._set_initial_parameters(rotations, translations) 

        # Alignment Parameters [n, f-2] f=num_subframes. n为image数量
        n = len(self)  
        self._nu = nn.Parameter(inverse_sigmoid( torch.linspace(1/(self.f-1), 1.0-(1/(self.f-1)), self.f-2)[None,:].repeat(n,1).cuda()).contiguous().requires_grad_(True) )
    
    def link_gaussian(self, gaussians:GaussianModel):
        """
        Register GaussianModel Object.
        """
        self.gaussians = gaussians 

    # 将曲线参数添加到gaussian的优化器中
    def add_training_setup(self, gaussians:GaussianModel, lr_dict:dict, args):
        """
        Extend gaussianmodel optimizer
        by adding pose_optimizer parameters.
        """
        
        # Delete previous curve parameters from optimizer. 
        # optimizer.state = {A:{...}, B:{...}, C:{...}}  A:value to params
        for group in gaussians.optimizer.param_groups:
            if "curve_" in group['name'] and group['params'][0] in gaussians.optimizer.state:
                del gaussians.optimizer.state[group['params'][0]]
        gaussians.optimizer.param_groups = [e for e in gaussians.optimizer.param_groups if 'curve_' not in e['name']]
        
        gaussians.optimizer.add_param_group({'params': self._rot.parameters(),   'lr': lr_dict['curve_rot'], 'name': 'curve_rot'})
        if hasattr(self, "_trans"):
            gaussians.optimizer.add_param_group({'params': self._trans.parameters(),   'lr': lr_dict['curve_trans'], 'name': 'curve_trans'})
        gaussians.optimizer.add_param_group({'params': [self._nu],  'lr': lr_dict['curve_alignment'], 'name': 'curve_alignment'})
        
        # if hasattr(self, "weight_function"):
        #     self.weight_function.training_setup(args)  
        if hasattr(self, "weight_function"):
            gaussians.optimizer.add_param_group({'params': self.weight_function.weight_gen.parameters(), 'lr': lr_dict['subframe_weight_lr'], 'name': 'weight_function'})
        
    def query(self, cam_idx:int, 
                    subframe_indice,
                    post_process=None, 
                    background="random"):
        """
        Main query method.
        Render a blurry view, and retrieve additional queried data.

        ARGUMENTS
        ---------
        - cam_idx: int
            camera index

        - subframe_indice: "all", list[int] or int
            If "all" (Default), render all subframes.
            If list(or iterable) of int, this indicates subframe indice.
            If int, this indicates the number of subframes to be rendered; indice are evenly-spaced.
        
        - post_process: None or Callable.
            Postprocess (e.g. gamma-correction) for blurry view. Do nothing if None.
        
        - background: "random" or torch.tensor[3]
            background color. random or color in [0.0, 1.0]
        
        RETURNS
        -------
        - retrieved: dictionary
             dictionary of answered query, whose keys are
            - 'blurred': synthesized blurry view. Post_process will be applied here.
            - 'gt': gt observation. (Default)
            - 'subframes': all subframe renderings.
            - 'render_pkgs': list of render_pkgs from 3DGS render function.
            - 'depths': all subframe depth renderings.
        """ 

        assert hasattr(self, "gaussians") and isinstance(self.gaussians, GaussianModel)

        gaussians = self.gaussians
        
        # Configure background color.
        if background == "random":
            bg = torch.rand(3,device=gaussians._xyz.device)
        else:
            bg = background
            

        # Generate sub-frame cams.
        if subframe_indice == "all":
            subframe_cams = self.get_trajectory(cam_idx)
        else:
            nu = self._sample_nu_from_alignment(cam_idx)   # nu is samples  [f]
            if isinstance(subframe_indice, int):
                if subframe_indice == 1:
                    # subfr_idx = [len(self)//2]
                    subfr_idx = [int(self.f)//2]
                # subfr_idx = torch.linspace(0,nu.shape[0]-1, subframe_indice, device=nu.device).long()
                else:
                    subfr_idx = []
                    subfr_idx.append(subframe_indice)
            else:
                subfr_idx : list = subframe_indice
            nu = nu[subfr_idx]
            subframe_cams = self.get_trajectory(cam_idx, nu)
            
        # Main code for render.
        render_pkg_subframes = []

        if self.USE_REF_CAM:
            ref_cam = self.original_cam[cam_idx]
            
            render_pkg_ref = gaussian_renderer.render(ref_cam, gaussians, bg)  # dict
            render_pkg_subframes.append(render_pkg_ref)
            #print(f"+++ref_depth_shape is {render_pkg_ref['depth'].shape}")
        else:
            render_pkg_ref = None   



        for cam in subframe_cams:
            render_pkg = gaussian_renderer.render(cam, gaussians, bg)
            render_pkg_subframes.append(render_pkg)   
        
        render_subframes  = torch.stack([render_pkg['render'] for render_pkg in render_pkg_subframes]) # [f,3,h,w], f is num_subframes.
        #print(f'++++++++++++{render_subframes.shape}')
        render_depth_subframes = torch.stack([render_pkg['depth'] for render_pkg in render_pkg_subframes])  # [f,1,h,w]
        #print(f'============{render_depth_subframes.shape}')

        if self.use_weight_generator and self.WG_FLAG:
            # 使用 MLP 生成权重
            num_subframes = self.f  # f
            subframe_index = torch.arange(num_subframes+1, device=render_subframes.device).float()  # [f]
            weights = self.weight_function(subframe_index[None, :]).squeeze(0)  # [f+1]
            # 对权重进行归一化（确保总和为 1）
            weights = weights / weights.sum()
            # 加权和计算最终结果
            blurred = (weights[:, None, None, None] * render_subframes).sum(dim=0)  # [3, h, w]
        else:
            blurred = render_subframes.mean(dim=0) # [3,h,w]

        blurred_depth = render_depth_subframes.mean(dim=0)
        # Return Values
        retrieved_dic = {}
 
        if post_process is not None:
            blurred = post_process(blurred)
        
        retrieved_dic['blurred'] = blurred
        retrieved_dic['blurred_depth'] = blurred_depth
        retrieved_dic['render_ref'] = render_pkg_ref['render'] if render_pkg_ref is not None else None
        retrieved_dic['gt'] = self.get_gt_image(cam_idx) # [3,h,w]
        retrieved_dic['gt_depth'] = self.get_gt_depth(cam_idx)
        retrieved_dic['gt_depth_reliable'] = self.get_gt_depth_reliable(cam_idx)
        retrieved_dic['gt_depth_mask'] = self.get_gt_depth_mask(cam_idx)
        retrieved_dic['subframes'] = render_subframes
        retrieved_dic['subframes_depths'] = torch.stack([render_pkg['depth'] for render_pkg in render_pkg_subframes])
        retrieved_dic['render_pkgs'] = render_pkg_subframes
    
        return retrieved_dic, render_pkg_ref

    def get_trajectory(self, idx, t=None):
        """
        idx: int
        t: None or torch.tensor of size [f (#_of_frames)]. 
           (tensor of) position on the trajectory in the range of [0,1].
           if None, sample from alignment parameter "t" of this model.
        RETURN
        ------
        list of MiniCam type objects (which can be used in rasterization later.)
        corresponding to camera idx.
        """

        # sample subframe c2w_rotations, c2w_translations.
        rot_interp, trans_interp = self._sample_c2w_from_nu(idx, t)

        # Convert to list of Minicam objects, and returns.
        return self._c2w_to_minicam(rot_interp, trans_interp, self.original_cam[idx])
    
    def _set_initial_parameters(self, rotations, translations):
        """
        set initial Bezier control points' parameters.

        ARGUMENTS
        ---------
        rotations: rotation part of c2w matrix [n, 3, 3]
        translations: camera origin (or equivalently, translation part of c2w matrix [n,3])
        """
        n = rotations.shape[0]

        if self.curve_type == "quarternion_cartesian":  # 四元数
            rot_params = roma.rotmat_to_unitquat(rotations) # [n,4]
            self._rot = BezierModel(rot_params, self.curve_order)
            self._trans = BezierModel(translations, self.curve_order, initial_noise=0.01)

        elif self.curve_type == "se3":
            # NOTE: transpose for torch3d convention
            c2w = torch.zeros(n,4,4).cuda()
            c2w[:,:3,:3] = rotations.transpose(-2,-1)
            c2w[:,3,:3] = translations
            c2w[:,3,3] = 1.0

            params = torch3d.se3_log_map(c2w) # pose transform to se3 [n,6]
            self._rot = BezierModel(params[:,3:], self.curve_order)
            self._trans = BezierModel(params[:,:3], self.curve_order)
        else:
            raise NotImplementedError
        
    def _sample_nu_from_alignment(self, idx): 
        
        
        device = self._nu.device
        
        nu_mid = torch.sigmoid(self._nu[idx]) # [f-2]  取某个相机轨迹曲线
        if self.curve_random_sample:
            nu_mid = nu_mid + torch.rand_like(nu_mid) / self.n_subframes - (1/(2*self.n_subframes)) # add some "uncertainty"
     

        # return nu_mid.sort().values # HARDCODING.
        return torch.cat([torch.zeros(1, device=device), nu_mid, torch.ones(1, device=device)]).clamp(0.0, 1.0).sort().values # [f]

    def _sample_c2w_from_nu(self, idx, nu=None): # 根据时间参数nu差值位姿
        """
        ARGUMENTS
        ---------
        idx: curve index.
        t: Tensor of shape [num_subframes,], ranging in [0.0,1.0] 

        RETURNS
        -------
        c2w_rotations: Tensor of shape [num_subframes, 3, 3] 
        c2w_translations: Tensor of shape [num_subframes, 3]
        """
        
        if nu is None:
            nu = self._sample_nu_from_alignment(idx)  # _nu对应采样点t  nu:[f]
        elif torch.is_tensor(nu):
            nu = nu.to(self.device)
        else:
            raise NotImplementedError

        
        if self.curve_type == "quarternion_cartesian":
            rot_quaternion = self._rot(nu,idx) # [f,4]
            rot_quaternion = rot_quaternion / rot_quaternion.norm(dim=1, keepdim=True) # [f,4]
            c2w_rotations = roma.unitquat_to_rotmat(rot_quaternion) # [f,3,3]
            c2w_translations = self._trans(nu,idx) # [f,3]

        elif self.curve_type == "se3":
            se3 = torch.cat([self._trans(nu,idx), self._rot(nu, idx)], dim=1) # [f,6]
            c2w = torch3d.se3_exp_map(se3) # [f,4,4]
            c2w_rotations = c2w[:,:3,:3].transpose(-2,-1)
            c2w_translations = c2w[:,3,:3]

        else:
            raise NotImplementedError
        return c2w_rotations, c2w_translations
        
    def _c2w_to_minicam(self, rots, transes, ref_cam:Camera):
        """
        given batch of rotation and translation in c2w poses,
        returns minicam object.

        ARGUMENTS
        ---------
        rots: [b,3,3]
        transes: [b,3]
        ref_cam: Camera or Minicam object. 
                 Additional attributes (znear, zfar, fov, etc...) will be duplicated from here.
        RETURNS
        -------
        list of minicam objects
        """

        minicam_list = []
        for i, (rot,trans) in enumerate(zip(rots,transes)): # c2w
            
            world_view_transform = torch.eye(4, device=self.device)
            world_view_transform[:3,:3] = rot # NOTE rot.T.T 
            world_view_transform[3,:3] = -trans@rot # NOTE: not [:3,3] for world-view transform.
            
            projection_matrix = ref_cam.projection_matrix
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            minicam_list.append(
                MiniCam(width=ref_cam.image_width,
                        height=ref_cam.image_height,
                        fovy=ref_cam.FoVy,
                        fovx=ref_cam.FoVx,
                        znear=ref_cam.znear,
                        zfar=ref_cam.zfar,
                        world_view_transform=world_view_transform,
                        full_proj_transform=full_proj_transform,)
            )
        
        return minicam_list
    
    def get_gt_image(self, idx):
        return self.original_cam[idx].original_image.cuda()
        
    def get_gt_depth(self, idx):
        if self.original_cam[idx].invdepthmap is not None:
            return self.original_cam[idx].invdepthmap.cuda()
        else:
            return None

    def get_gt_depth_reliable(self,idx):
        return self.original_cam[idx].depth_reliable

    def get_gt_depth_mask(self,  idx):
        if self.original_cam[idx].depth_mask is not None:
            return self.original_cam[idx].depth_mask.cuda()
        else:
            return None

    def __len__(self):
        return len(self._rot)

    @property
    def device(self):
        return self._rot.device
    
    def is_optimizing(self):
        return self._rot._control_points.requires_grad
    
    def alternate_optimization(self):
        """
        Stop optimizing if it was doing. Start optimizing if optimizing process was stopped.
        """
        new_state = not self.is_optimizing()
        self.WG_FLAG = True
        
        print("Curve gradient:" , "[On]" if new_state else "[Off]")
        for optimizable in [self._rot, self._trans, self._nu]:
            optimizable.requires_grad_(new_state)
        
    @torch.no_grad()
    def get_middle_cams(self):
        """
        get list of "middle" from the trajectory.
        """
        cams = []
        for i in range(len(self)):
            nu = self._sample_nu_from_alignment(i)
            mid_idx = nu.shape[0]//2
            nu_mid = nu[mid_idx: mid_idx+1]
            cam = self.get_trajectory(i,nu_mid)[0]
            cams.append(cam)
        return cams
    
    
    def save(self, state_dict_path:str):
        """
        Save camera motion parameters.
        """
        
        assert(state_dict_path.endswith(".pth"))
        
        sdict = {"rot": self._rot.state_dict(),
                 "trans": self._trans.state_dict(),
                 "nu": self._nu}

        torch.save(sdict, state_dict_path)
        
        print("[SAVED] Camera Motion")

    def load(self, path:str):
        """
        Load camera motion parameters.
        """

        if path.endswith(".pth"):
            state_dict_path = path
        else:
            state_dict_path = os.path.join(path,"cm.pth")

        sdict = torch.load(state_dict_path)
        self._rot.load_state_dict(sdict["rot"])
        self._trans.load_state_dict(sdict["trans"])
        self._nu = sdict['nu']
        print("[LOADED] Camera Motion")

     # Add new loss for smoothness of _control_points
    def compute_smoothness_loss(self):
        """
        计算控制点的平滑性损失。
        """
        rot_loss = self._compute_pairwise_loss(self._rot._control_points)
        trans_loss = self._compute_pairwise_loss(self._trans._control_points)
        return rot_loss + trans_loss

    def _compute_pairwise_loss(self, control_points):
        """
        计算相邻控制点之间的平滑性损失。
        """
        diff = control_points[:, 1:, :] - control_points[:, :-1, :]  # 相邻控制点的差
        return (diff ** 2).mean()  # L2 损失