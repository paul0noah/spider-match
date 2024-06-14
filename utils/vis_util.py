import scipy.io
from vedo import *
import copy
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
import cv2
import os
import potpourri3d as pp3d
from utils.vis_helpers import *

##
#
# Settings
#
##

# plot look
texture_name = 'texture_8k_bnw_nn_tp30.png'
screenshot_scale = 3
export_both_pairs = False
interiorcolor_as_exterior = True
back_face_color = 0.45 * np.array([1, 1, 1])
color_mode = 'unitbox'
invert_color_channel = 'r'  # r, b, g, rb, rg, bg, rbg
plot_zoom = 1
mesh_opacity = 1
plot_shadows = True
plot_texture = False
plot_color = True
color_bias = [1, 0.75, 1]
color_scale = 0.9
color_offset = 0.1
settings.use_depth_peeling = True
light_intensity_mult = 0.9
settings.use_depth_peeling = True
mesh_lighting = 'default'  # ['default', 'metallic', 'plastic', 'shiny', 'glossy', 'ambient', 'off']

# morph settings
num_frames_img = 11


##
#
# End Settings
#
##


def get_cams_and_rotation(dir):
    if 'faust' in dir.lower():
        cam = dict(
            position=(-1.85922, -4.78140, 0.689328),
            focal_point=(0.192544, 4.77379e-3, 0.0127248),
            viewup=(0.0724348, 0.109097, 0.991388),
            distance=5.25119,
            clipping_range=(4.38228, 6.36775),
        )
        camSingle = dict(
            position=(-0.900626, -2.17409, 0.297233),
            focal_point=(0.0571778, 0.0596532, -9.57414e-3),
            viewup=(0.0710136, 0.105782, 0.991850),
            distance=2.44972,
            clipping_range=(1.75096, 3.18201),
        )
        rotationX = [85, 0, 0]
        rotationY = rotationX
        return [cam, camSingle, rotationX, rotationY]
    if 'smal' in dir.lower():
        cam = dict(
            position=(-2.41391, -5.72555, 0.887058),
            focal_point=(0.0687253, 0.0657214, 0.0683672),
            viewup=(0.0724348, 0.109097, 0.991388),
            distance=6.35394,
            clipping_range=(4.94716, 8.20514),
        )
        camSingle = dict(
            position=(-1.80646, -4.83287, 0.449118),
            focal_point=(-7.40232e-3, 0.0780391, -0.0214043),
            viewup=(0.0465298, 0.0783672, 0.995838),
            distance=5.25119,
            clipping_range=(3.88061, 7.16671),
        )
        rotationX = [-100, 0, 90]
        rotationY = [-100, 0, -90]
        return [cam, camSingle, rotationX, rotationY]
    if 'dt4' in dir.lower():
        cam = dict(
            position=(-2.22250, -5.82165, 0.787677),
            focal_point=(0.260137, -0.0303834, -0.0310133),
            viewup=(0.0724348, 0.109097, 0.991388),
            distance=6.35394,
            clipping_range=(5.30777, 7.89410),
        )
        camSingle = dict(
            position=(-2.37678, -5.75643, 0.781225),
            focal_point=(0.105854, 0.0348434, -0.0374659),
            viewup=(0.0724348, 0.109097, 0.991388),
            distance=6.35394,
            clipping_range=(5.30777, 7.89410),
        )
        rotationX = [0, 0, 0]
        rotationY = [0, 0, 0]
        return [cam, camSingle, rotationX, rotationY]

##
#
# Helpers
#
##

def get_uv(points, normal, d):
    assert (abs(np.linalg.norm(normal) - 1) < 1e-4)
    # choose principal component which has smallest angle to camera normal
    z_axis = np.array([0, 0, 1])
    pca = PCA(n_components=3)
    pca = PCA(n_components=3)
    pca.fit(points)
    angles = np.arccos(np.clip(np.dot(normal, pca.components_.transpose()), -1.0, 1.0))
    angles2 = np.arccos(np.clip(np.dot(-normal, pca.components_.transpose()), -1.0, 1.0))
    all_angles = np.array([angles, angles2])
    proj_comp = pca.components_[all_angles.min(axis=0).argmin(), :]
    normal = proj_comp

    # project points on plane
    uv = points - (np.reshape(np.sum(points * normal, axis=1), [points.shape[0], 1]) + d) * normal

    # compute angle with z axis and rotate such that z component becomes zero
    ax = np.cross(normal, z_axis)
    angle = np.arccos(np.clip(np.dot(normal, z_axis), -1.0, 1.0))
    r = R.from_rotvec(ax * angle)
    uv_rot = uv @ r.as_matrix().transpose()
    uv_rot[:, 2] = 0

    # make sure principal components of x-y data align with xy axis (so that texture is not at an angle)
    pca = PCA(n_components=1)
    pca.fit(uv_rot)
    main_ax = pca.components_ / np.linalg.norm(pca.components_)
    y_axis = np.array([0, 1, 0])
    angle = np.arctan2(np.dot(np.cross(main_ax, y_axis), z_axis), np.dot(main_ax, y_axis))
    # angle = np.arccos(np.clip(np.dot(main_ax, y_axis), -1.0, 1.0))
    if abs(angle) > np.pi / 2:
        angle = angle + np.sign(angle) * np.pi
    r = R.from_rotvec(z_axis * angle)
    uv_rot_rot = uv_rot @ r.as_matrix().transpose()
    uv = uv_rot_rot[:, :2]
    uv = uv - uv.min(axis=0)
    uv = uv / uv.max()
    return uv


def smooth_text_transfer(VX, FX, VY, FY, uv_x, p2p):
    assert (len(p2p) == len(VY))
    _, evecs_x, evecs_trans_x, _ = laplacian_decomposition(VX, FX, k=150)
    _, evecs_y, evecs_trans_y, _ = laplacian_decomposition(VY, FY, k=150)
    Cxy = evecs_trans_y @ evecs_x[p2p]
    Pyx = evecs_y @ Cxy @ evecs_trans_x
    return Pyx @ uv_x


def get_rgb_cmap(mesh, offset_low=None, scale=None, color_bias=[1, 1, 1], invert='rg', mode='geodist'):
    mode = color_mode  # 'unitbox'
    invert = invert_color_channel  # ''
    if mode == 'unitbox':
        cmap = np.array(mesh.points())
        minx = cmap[:, 0].min()
        miny = cmap[:, 1].min()
        minz = cmap[:, 2].min()
        maxx = cmap[:, 0].max()
        maxy = cmap[:, 1].max()
        maxz = cmap[:, 2].max()
        r = (cmap[:, 0] - minx) / (maxx - minx)
        g = (cmap[:, 1] - miny) / (maxy - miny)
        b = (cmap[:, 2] - minz) / (maxz - minz)
        cmap = np.stack((r, g, b), axis=-1)
    elif mode == 'geodist':
        meshS = Mesh([mesh.points(), np.array(mesh.faces())])
        meshS.decimate(n=100)
        D = geodesic_distmat_dijkstra(meshS.points(), np.array(meshS.faces()))
        infty_idx = np.isinf(D[:, 0])
        D[infty_idx, :] = 0
        D[:, infty_idx] = 0
        indices = farthest_point_sampling_distmat(D, 3, random_init=False, verbose=False)
        indices = knn_search(meshS.points()[indices], mesh.points())
        cmap = np.array(mesh.points())

        solver = pp3d.MeshHeatMethodDistanceSolver(mesh.points(), np.array(mesh.faces()))
        cmap[:, 0] = solver.compute_distance(indices[0])
        cmap[:, 1] = solver.compute_distance(indices[1])
        cmap[:, 2] = solver.compute_distance(indices[2])

    elif mode == 'laplacian':
        _, evecs, _, _ = laplacian_decomposition(mesh.points(), np.array(mesh.faces()), k=3)
        cmap = evecs
    else:
        raise Exception("Colormode not supported get_rgb_cmap")
    cmap = cmap - cmap.min(axis=0)
    cmap = cmap / cmap.max(axis=0)

    if 'r' in invert.lower():
        cmap[:, 0] = abs(cmap[:, 0] - 1)
    if 'g' in invert.lower():
        cmap[:, 1] = abs(cmap[:, 1] - 1)
    if 'b' in invert.lower():
        cmap[:, 2] = abs(cmap[:, 2] - 1)

    if scale is not None:
        cmap = cmap * scale

    if offset_low is not None:
        cmap = cmap + offset_low
        cmap = cmap / cmap.max(axis=0)
        if scale is not None:
            cmap = cmap * scale

    cmap = cmap * np.array(color_bias)
    return (cmap * 255).round()


def save_png_rm_bg(filename, im, bg_color_min=[0, 255, 100], bg_color_max=None, rgb=True):
    if bg_color_max is None:
        bg_color_max = bg_color_min
    min = np.array(bg_color_min, np.uint8)
    max = np.array(bg_color_max, np.uint8)
    mask = cv2.inRange(im, min, max)

    # create alpha channel
    alpha = (im[:, :, 0] * 0 + 255)

    # Make all pixels in mask white
    alpha[mask > 0] = 0

    # add alpha channel
    if rgb:
        im = cv2.merge((im[:, :, 0], im[:, :, 1], im[:, :, 2], alpha))
    else:
        im = cv2.merge((im[:, :, 2], im[:, :, 1], im[:, :, 0], alpha))

    # crop to visible part
    mask = im[:, :, 3] != 0.
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    im = im[y0:y1, x0:x1, :]

    # write to file
    cv2.imwrite(filename, im)


def get_lights(light_color=[255, 255, 245], offset=[0, 0, 0], intensity_mult=1):
    # Add light sources at the given positions
    # (grab the position and color of the arrow object)
    orig = np.array([0, 0, 0]) + np.array(offset)
    phl = Arrow(np.array([0.1, 0.1, 10]) + offset, orig, c=light_color).scale(0.2)
    pfl = Arrow(np.array([1.5, 0.1, 0.3]) + offset, orig, c=light_color).scale(0.2)
    pbl = Arrow(np.array([-1.5, 0.1, 0.3]) + offset, orig, c=light_color).scale(0.2)
    prl = Arrow(np.array([0.1, -1.5, 0.3]) + offset, orig, c=light_color).scale(0.2)
    pll = Arrow(np.array([0.1, 1.5, 0.3]) + offset, orig, c=light_color).scale(0.2)
    hl = Light(phl, intensity=0.7 * intensity_mult, angle=180, )
    rl = Light(pfl, intensity=0.6 * intensity_mult, angle=180, )
    ll = Light(pbl, intensity=0.6 * intensity_mult, angle=180, )
    bl = Light(pll, intensity=0.6 * intensity_mult, angle=180, )
    fl = Light(prl, intensity=1 * intensity_mult, angle=180, )
    lights = [hl, fl, bl, ll, rl]
    return lights


def get_mesh(mesh_name, mesh_opacity=0.8, mesh_lighting='default', rotation=[0, 0, 0], offset=[0, 0, 0], scale=0):
    offset = np.array(offset)
    # styles = ['default', 'metallic', 'plastic', 'shiny', 'glossy', 'ambient', 'off']
    mesh = Mesh(mesh_name, c='k7', alpha=mesh_opacity).lighting(mesh_lighting)
    if scale == 0:
        bnd = np.array(mesh.bounds())
        scale = 1 / max(bnd[1] - bnd[0], bnd[3] - bnd[2], bnd[5] - bnd[4])
    mesh.scale(scale)
    mesh.pos(offset)
    mesh.rotate_x(rotation[0])
    mesh.rotate_y(rotation[1])
    mesh.rotate_z(rotation[2])
    mesh.color(0.65 * np.array([1, 1, 1]))
    # mesh.phong()
    # mesh2.phong()
    return mesh, scale


def color_and_texturize_mesh(mesh, cmap, tcoord=[], plot_color=True, plot_texture=False,
                             texture_name='texture_8k_bnw_nn_tp30.png'):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    texture_name = os.path.join(dir_path, 'textures', texture_name)
    if plot_texture:
        repeat = False
        interpolate = False
        if len(tcoord) != 0:
            mesh.texture(tname=texture_name, repeat=repeat, tcoords=tcoord, interpolate=interpolate)
        else:
            mesh.texture(tname=texture_name, repeat=repeat, interpolate=interpolate)
    if plot_color:
        mesh.pointdata['vertex_colors'] = np.array(cmap).astype(np.uint8)
        mesh.pointdata.select('vertex_colors')
        mesh.pointcolors = np.array(cmap).astype(np.uint8)
        # per triangle coloring works somehwo like this
        # mesh.celldata['vertex_colors'] = np.array(get_rgb_cmap(mesh)).astype(np.uint8)
        # mesh.celldata.select('vertex_colors')
    if not plot_color and not plot_texture:
        mesh.color([128, 128, 128], 1)
        mesh.opacity(0.7)
    return mesh


def get_mesh_shadow(mesh, offset=[0, 0, 0], plane_normal=(0, 0, 1), direction=[0.1, -1.8, 3]):
    shadow = []
    shad_col = np.array([0.8, 0.8, 0.8])
    min_z = mesh.points().min(axis=0)[2]
    # mesh.add_shadow(plane='z', point=min_z, alpha=1, c=shad_col, culling=0.9,)
    plane = Plane(pos=np.array([0, 0, min_z]) + np.array(offset), normal=plane_normal, s=[7, 7]).alpha(0.2)
    shad = mesh.clone().project_on_plane(plane, direction=-np.array(direction) + np.array(offset))
    shad.c(shad_col).alpha(1).lighting("off").use_bounds(False)
    shadow = shad
    return shadow


def save_plot_as_transparent_png(plt, out_filename):
    # if not out_filename is None:
    bg_col = [255, 255, 255]
    plt.background(c1=bg_col)
    save_png_rm_bg(out_filename, plt.screenshot(asarray=True, scale=screenshot_scale), bg_color_min=bg_col)


def save_png(plt, imageName, meshX, meshY, plot_color, tempshapes, camBoth, camSingle, plot_zoom, meshYShad, meshXShad,
             kptsX=[], kptsY=[]):
    rgb = False
    plt.show(tempshapes, interactive=False, camera=camBoth, resetcam=True, zoom=plot_zoom)
    bg_col = [255, 255, 255]
    plt.background(c1=bg_col)
    if export_both_pairs:
        save_png_rm_bg(imageName, plt.screenshot(asarray=True, scale=screenshot_scale), bg_color_min=bg_col, rgb=rgb)

    img_name = imageName.split('.')

    # save X
    plt.remove(meshY, meshYShad)
    if kptsY:
        plt.remove(kptsY)
    elif interiorcolor_as_exterior:
        meshX.backface_culling(True)
        meshX_interior = Mesh([meshX.points(), meshX.faces()])
        meshX_interior.backcolor(back_face_color)
        meshX_interior.frontface_culling(True)
        plt.add(meshX_interior)
    plt.show(interactive=False, camera=camSingle, zoom=plot_zoom)
    settings.screenshot_transparent_background = False
    save_png_rm_bg(img_name[0] + '_M.' + img_name[1], plt.screenshot(asarray=True, scale=screenshot_scale),
                   bg_color_min=bg_col,
                   rgb=rgb)

    # save Y
    # meshY.pos(np.array([0, 0, 0]))
    plt.remove(meshX, meshXShad)
    if kptsX:
        plt.remove(kptsX)
    elif interiorcolor_as_exterior:
        plt.remove(meshX_interior)
    if kptsY:
        plt.show(meshY, meshYShad, kptsY, interactive=False, camera=camSingle, zoom=plot_zoom)
    elif interiorcolor_as_exterior:
        meshY.backface_culling(True)
        meshY_interior = Mesh([meshY.points(), meshY.faces()])
        meshY_interior.backcolor(back_face_color)
        meshY_interior.frontface_culling(True)
        plt.show(meshY, meshYShad, meshY_interior, interactive=False, camera=camSingle, zoom=plot_zoom)
    save_png_rm_bg(img_name[0] + '_N.' + img_name[1], plt.screenshot(asarray=True, scale=screenshot_scale),
                   bg_color_min=bg_col,
                   rgb=rgb)

##
#
# End Helpers
#
##

##
#
# PLOT FUNCTIONS
#
##

def plot_match(vx, fx, vy, fy, matching, cam, image_name='', offsetX=[0, 0, 0], offsetY=[0, 0, 0], rotationShapeX=[0, 0, 0], rotationShapeY=[0, 0, 0]):

    meshX, scaleX = get_mesh((vx, fx), mesh_opacity, mesh_lighting, rotationShapeX, offsetX)
    cmap = get_rgb_cmap(meshX, offset_low=color_offset, scale=color_scale, color_bias=color_bias)

    uv_x = []
    if plot_texture:
        if not cam:
            normal = np.array([1, 1, 0])
            normal = normal / np.linalg.norm(normal)
            d = 0
        else:
            normal = np.array(cam['position']) - np.array(cam['focal_point'])
            normal = normal / np.linalg.norm(normal)
            d = - np.matmul(normal.transpose(), (np.array(cam['position'])))
        uv_x = get_uv(meshX.points(), normal, d)

    meshX = color_and_texturize_mesh(meshX, cmap, uv_x, plot_color, plot_texture)
    bndX = np.array(meshX.bounds())

    meshY, _ = get_mesh((vy, fy), mesh_opacity, mesh_lighting, rotationShapeY, offsetY, scaleX)
    bndY = np.array(meshY.bounds())
    # move z on same level
    meshY.pos(offsetY - np.array([0, 0, bndY[4] - bndX[4]]))

    uv_y = []
    if plot_texture:
        uv_y = smooth_text_transfer(meshX.points(), np.array(meshX.faces()), meshY.points(),
                                    np.array(meshY.faces()), uv_x, matching.flatten())
    cmapY = 0 * vy
    cmapY[matching[:, 1]] = cmap[matching[:, 0]]
    meshY = color_and_texturize_mesh(meshY, cmapY, uv_y, plot_color, plot_texture)

    of = False
    if image_name:
        of = True
    plt = Plotter(bg=[255, 255, 255], offscreen=of)
    lights = get_lights(intensity_mult=light_intensity_mult)

    outtimes = np.linspace(0, 1, num=11, endpoint=True)
    # plt.move_camera(cameras=[cam1, cam2], t=0, smooth=True, output_times=outtimes)
    # plt.play()

    shapes = []
    for light in lights:
        shapes.append(light)
    tempshapes = shapes
    tempshapes.append(meshX)
    tempshapes.append(meshY)
    if plot_shadows:
        meshXShad = get_mesh_shadow(meshX)
        meshYShad = get_mesh_shadow(meshY)
        tempshapes.append(meshXShad)
        tempshapes.append(meshYShad)

    if not image_name:
        if interiorcolor_as_exterior:
            meshX_interior, meshY_interior = Mesh([meshX.points(), meshX.faces()]), Mesh(
                [meshY.points(), meshY.faces()])
            meshX.backface_culling(True)
            meshY.backface_culling(True)
            meshX_interior.frontface_culling(True).backcolor(back_face_color)
            meshY_interior.frontface_culling(True).backcolor(back_face_color)
            plt.add(meshX_interior, meshY_interior)
        plt.show(tempshapes, interactive=True, camera=cam, resetcam=True, zoom=plot_zoom)
    else:
        save_png(plt, image_name, meshX, meshY, plot_color, tempshapes, cam, cam, plot_zoom, meshYShad,
                 meshXShad, )