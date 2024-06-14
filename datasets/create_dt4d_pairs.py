import random
import os
import igl
import open3d as o3d
import numpy as np
import pymeshfix

def get_genus(v, f):
    e = igl.edges(f)
    return int(round(-0.5 * (v.shape[0] - e.shape[0] + f.shape[0] -2)))

def read_shape(file, as_cloud=False):
    if as_cloud:
        verts = np.asarray(o3d.io.read_point_cloud(file).points)
        faces = None
    else:
        mesh = o3d.io.read_triangle_mesh(file)
        verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    return verts, faces

def shape_loader(filename1, filename2, shape_loader_opts):
    vert_np_x, face_np_x = read_shape(filename1)
    vert_np_y, face_np_y = read_shape(filename2)
    vert_np_x -= np.mean(vert_np_x, axis=0)
    vert_np_y -= np.mean(vert_np_y, axis=0)

    VX_orig = vert_np_x
    FX_orig = face_np_x
    VY_orig = vert_np_y
    FY_orig = face_np_y
    VX, FX = pymeshfix.clean_from_arrays(VX_orig, FX_orig)
    VY, FY = pymeshfix.clean_from_arrays(VY_orig, FY_orig)
    nfacesX = min(shape_loader_opts['num_faces'], len(FX_orig)) +3
    nfacesY = min(shape_loader_opts['num_faces'], len(FY_orig))

    if shape_loader_opts['use_qslim']:
        _, VX, FX, _, _ = igl.qslim(VX, FX, nfacesX)
        _, VY, FY, _, _ = igl.qslim(VY, FY, nfacesY)
    else:
        _, VX, FX, _, _ = igl.decimate(VX, FX, nfacesX)
        _, VY, FY, _, _ = igl.decimate(VY, FY, nfacesY)

    idx_vx_in_orig = []
    idx_vy_in_orig = []

    return VX_orig, FX_orig, VX, FX, idx_vx_in_orig, VY_orig, FY_orig, VY, FY, idx_vy_in_orig


dataset = 'DT4D_r/off'
files = []
with open(os.path.join(dataset, '..', 'test.txt')) as testset:
    for line in testset:
        files.append(line.strip())

# randomise the files list
random.shuffle(files)

max_num_pairs = 100
max_num_per_shape = 3

# intra
with open('pairs_intraclass.txt', 'a') as the_file:
    num_pairs = 0
    for idxX in range(0, len(files)):
        num_per_shape = 0
        if 'pumpkinhulk' in files[idxX].split('/')[0]:
            # skip, see attentive fmaps implementation
            continue
        for idxY in range(idxX+1, len(files)):
            if files[idxX].split('/')[0] not in files[idxY]:
                continue
            if 'pumpkinhulk' in files[idxY].split('/')[0]:
                # skip, see attentive fmaps implementation
                continue
            if num_pairs >= max_num_pairs:
                break
            if num_per_shape >= max_num_per_shape:
                break


            filename1 = os.path.join(dataset, files[idxX] + '.off')
            filename2 = os.path.join(dataset, files[idxY] + '.off')
            sl_opts = {"use_qslim": False, "num_faces": 450}
            _, _, VX, FX, _, _, _, VY, FY, _ = shape_loader(filename1, filename2, sl_opts)
            gX = get_genus(VX, FX)
            gY = get_genus(VY, FY)
            if gX != gY:
                continue
            name = files[idxX].split('/')[1] + '-' + files[idxY].split('/')[1]
            the_file.write(name + '\n')
            num_pairs = num_pairs + 1
            num_per_shape = num_per_shape + 1

    print("Num Intraclass " + str(num_pairs))

random.shuffle(files)
cc_pairs = os.listdir(dataset + '/../corres/cross_category_corres/')
# inter
with open('pairs_interclass.txt', 'a') as the_file:
    num_pairs = 0
    for idxX in range(0, len(files)):
        num_per_shape = 0
        if 'pumpkinhulk' in files[idxX].split('/')[0]:
            # skip, see attentive fmaps implementation
            continue
        for idxY in range(idxX+1, len(files)):
            if files[idxX].split('/')[0] in files[idxY]:
                continue
            if 'pumpkinhulk' in files[idxY].split('/')[0]:
                # skip, see attentive fmaps implementation
                continue
            if not files[idxX].split('/')[0] + '_' + files[idxY].split('/')[0] + '.vts' in cc_pairs:
                if not files[idxY].split('/')[0] + '_' + files[idxX].split('/')[0] + '.vts' in cc_pairs:
                    continue
            if num_pairs >= max_num_pairs:
                break
            if num_per_shape >= max_num_per_shape:
                break

            filename1 = os.path.join(dataset, files[idxX] + '.off')
            filename2 = os.path.join(dataset, files[idxY] + '.off')
            sl_opts = {"use_qslim": False, "num_faces": 450}
            _, _, VX, FX, _, _, _, VY, FY, _ = shape_loader(filename1, filename2, sl_opts)
            gX = get_genus(VX, FX)
            gY = get_genus(VY, FY)
            if gX != gY:
                continue
            name = files[idxX].split('/')[1] + '-' + files[idxY].split('/')[1]
            the_file.write(name + '\n')
            num_pairs = num_pairs + 1
            num_per_shape = num_per_shape + 1

    print("Num Interclass " + str(num_pairs))
