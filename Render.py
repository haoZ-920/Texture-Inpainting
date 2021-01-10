import os
import torch
import matplotlib.pyplot as plt
#from skimage.io import imread

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
# from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    TexturesVertex
)

# add path for demo utils functions
import sys
import os
sys.path.append(os.path.abspath(''))
import numpy as np
#from plot_image_grid import image_grid
import cv2
from smplx import build_layer as build_body_model
from pytorch3d.renderer.cameras import PerspectiveCameras

import random
import pickle


#from render_textured_meshes import read_params, read_displacement, read_txt, save_obj

class Render:
    def __init__(self, img_size ,batch_size, device, tex_path = None ):
        R = torch.from_numpy(np.expand_dims(np.array(
                                        [[1.0,0.0,0.0]
                                       ,[0.0,1.0,0.0]
                                       ,[0.0,0.0,1.0]]), axis=0)).type(torch.float)
        T = torch.from_numpy(np.expand_dims(np.array([0.0,0.0,0.0]),axis=0)).type(torch.float) #0.0,0.2,2.3

        width = img_size[0]#1080.0
        height = img_size[1]#1080.0

        focal = torch.from_numpy(np.expand_dims(np.array([width,width]),axis=0)).type(torch.float)
        center = torch.from_numpy(np.expand_dims(np.array([width/2,height/2]),axis=0)).type(torch.float)
        Size = torch.from_numpy(np.expand_dims(np.array([width,height]),axis=0)).type(torch.float)

        cam = PerspectiveCameras(device=device, R=R, T=T, focal_length=focal, principal_point=center, image_size=Size)
        # Define the settings for rasterization and shading.
        raster_settings = RasterizationSettings(
            image_size=int(width),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Place a point light in front of the object.
        lights = PointLights(device=device, location=[[10.0, 10.0, 10.0]])
        self.lights = lights

        cam_ = cam#cameras
        # Create a phong renderer by composing a rasterizer and a shader.
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cam_,#cam,#cameras,#
                raster_settings=raster_settings
            ),

            shader=SoftPhongShader(
                device=device,
                cameras=cam_,#cam,#cameras,#
                lights=lights
            )
        )

        f = np.loadtxt('./smplx_f_ft_vt/smplx_f.txt')
        ft = np.loadtxt('./smplx_f_ft_vt/smplx_ft.txt')
        vt = np.loadtxt('./smplx_f_ft_vt/smplx_vt.txt')

        self.fs = torch.from_numpy(np.expand_dims(f, axis=0)).repeat([batch_size,1,1]).type(torch.long).to(device)
        self.fts = torch.from_numpy(np.expand_dims(ft, axis=0)).repeat([batch_size,1,1]).type(torch.long).to(device)
        self.vts = torch.from_numpy(np.expand_dims(vt, axis=0)).repeat([batch_size,1,1]).type(torch.float).to(device)

        self.device = device
        self.batch_size = batch_size

        if tex_path is not None:
            tex = np.expand_dims(cv2.imread(tex_path ), axis=0)
            texture_image = torch.from_numpy(tex/255.0).type(torch.float).to(self.device) * 2.0
            self.texture = TexturesUV(texture_image, self.fts, self.vts)

    def load_texture(self, tex):
        #tex = np.expand_dims(cv2.imread(tex_path ), axis=0)

        tex = tex.permute(0,2,3,1) # CxHxW -> HxWxC
        tex = tex[:,:,:,[2,1,0]] # RGB -> BGR
        texture_image = tex.type(torch.float).to(self.device) * 2.0 # torch.from_numpy(tex/255.0)
        self.texture = TexturesUV(texture_image, self.fts, self.vts)
        pass


    def get_renderd_tensor(self, verts):
        vert = verts.clone()

        vert[:, :, 2] = vert[:, :, 2] * -1

        mesh = Meshes(vert, self.fs)
        mesh.textures = self.texture
        mesh = mesh.to(self.device)

        image = self.renderer(mesh, lights=self.lights)

        image = torch.flip(image, [2])

        # for idx in range(self.batch_size):
        #     img = image.detach().cpu().numpy()[idx, :, :, :3]
        #     cv2.imwrite("./results/test{}.png".format(idx), img * 255)

        return image




    def get_renderd(self, verts):
        verts[:, :, 2] = verts[:, :, 2] * (-1.0)

        mesh = Meshes(verts, self.fs)
        mesh.textures = self.texture
        mesh = mesh.to(self.device)

        image = self.renderer(mesh, lights=self.lights)
        img = image.detach().cpu().numpy()[0,:,:,:3]
        img = cv2.flip(img, 1)

        #cv2.imwrite("./output/renderd.png", img * 255)

        #img = torch.from_numpy(np.expand_dims(img, axis=0)).to(self.device)
        return img

    def get_silhouette(self, verts):

        verts[:, :, 2] = verts[:, :, 2] * (-1.0)

        black_image = torch.from_numpy(np.zeros([1,1080,1080,3])).type(torch.float).to(self.device)  # mesh.textures.maps_padded()
        mesh = Meshes(verts, self.fs)
        mesh.textures = TexturesUV(black_image, self.fts, self.vts)
        mesh = mesh.to(self.device)

        silhouette = self.renderer(mesh, lights=self.lights)
        silhouette_img = silhouette.detach().cpu().numpy()[0, :, :, :3]
        silhouette_img = cv2.flip(silhouette_img, 1)  # cv2.cvtColor(, cv2.COLOR_BGR2GRAY )
        silhouette_img = np.any(silhouette_img < 1, axis=-1)

        #cv2.imwrite("./output/renderd.png", silhouette_img * 255)
        silhouette_img = torch.from_numpy(np.expand_dims(silhouette_img, axis=0)).to(self.device)
        return silhouette_img


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    random.seed(0)

    base_path = './Fit_result'  #   20200923_vrc_test_data1
    out_path = './output_singleview'

    save_img_path = out_path + '/train_img/'# _c
    save_param_path = out_path + '/train_param/' # _c

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(save_img_path):
        os.mkdir(save_img_path)
    if not os.path.exists(save_param_path):
        os.mkdir(save_param_path)

    Dirs = sorted(os.listdir(base_path))

    model_cfg = {
        'gender':'neutral',
        'use_face_contour': True,

    }
    model_path = "./models/"
    smplx_model = build_body_model(model_path, "smplx", **model_cfg )

    def read_params():
        pass
    def read_displacement():
        pass
    for idx in range(len(Dirs)):
        # if idx > 1:
        #    break
        print( idx )
        #r = random.randint(0, pose_sequence.shape[0]-1)
        param_path = base_path + '/' + Dirs[idx] + '/para.txt'  # + 'para_{}'.format(dir_name.split('_')[-1]) + '.txt' '20180509095324947'
        displacement_path = base_path + '/' + Dirs[idx] + '/Displacement.txt'
        tex_path = base_path + '/' + Dirs[idx] + '/tex.png'

        pose, beta = None, None#read_params(param_path)
        displacement = None#read_displacement(displacement_path)

        rend = Render(tex_path, device)
        pose[0] = [0,0,0]
        r = [0.] + sorted(np.random.rand(9))
        for jIdx in range(10):

            pose = np.zeros([55,3])
            #pose[0:22,:] = pose_sequence[r,0:22,:]
            #displacement = np.zeros([10475,3])
            #beta = np.zeros([20])
            pose_c = pose.copy()
            pose_c[0] += [0, 3.1415926 * 2 * r[jIdx], 0]

            #ipose = torch.from_numpy(np.expand_dims(pose, axis=0)).type(torch.float)
            ibeta = torch.from_numpy(np.expand_dims(beta, axis=0)).type(torch.float)
            idisplacement = torch.from_numpy(np.expand_dims(displacement, axis=0)).type(torch.float)

            #np.save(save_param_path + Dirs[idx] + '.npy', param)
            #param_ = np.load(save_param_path + Dirs[idx] + '.npy', allow_pickle=True) #

            pose_matrix = []
            #for p in range(pose.shape[0]):
            pose_matrix.append([cv2.Rodrigues(pose_v)[0] for pose_v in pose_c])
            pose_matrix = np.array(pose_matrix)

            ipose_matrix = torch.from_numpy(pose_matrix).type(torch.float) # np.expand_dims(pose_matrix, axis=0)

            trans = np.array([0.0,0.4,-2.3]) #+ ( np.random.rand(3) -0.5 ) * 2.0 / 30.0

            merged_params = {
                'transl':torch.from_numpy(np.expand_dims(trans,axis=0)),
                'global_orient':ipose_matrix[:,0],
                'body_pose':ipose_matrix[:,1:22],
                'left_hand_pose':ipose_matrix[:,25:40],
                'right_hand_pose':ipose_matrix[:,40:55],
                'jaw_pose':ipose_matrix[:,22],
                'leye_pose':ipose_matrix[:,23],
                'reye_pose':ipose_matrix[:,24],
                'betas':ibeta[:,0:10],
                'expression':ibeta[:,10:20],
                'Displacements':idisplacement #/2.0
            }

            body_model_output = smplx_model(get_skin=True, return_shaped=True, **merged_params)

            verts = body_model_output['vertices'].type(torch.float)
            joints = body_model_output['joints'].type(torch.float)
            v = verts.detach().cpu().numpy().squeeze().copy()
            j = joints.detach().cpu().numpy().squeeze().copy()

            img = rend.get_renderd(verts)

            param = {
                'pose':pose,
                'betas':beta[:10,],
                'expression': beta[10:,],
                'translation':trans,
                'displacement':displacement,
                'keypoints_3d':j,
                'img_names':Dirs[idx],
                'vertices':v,
            }
            with open(save_param_path + Dirs[idx] + '_{}.pkl'.format(jIdx), 'wb') as file:
                pickle.dump(param, file, protocol=2)
            cv2.imwrite(save_img_path + Dirs[idx] + "_{}.png".format(jIdx), img * 255)

        #cv2.imwrite("./output/renderd.png", img * 255)


    pass

