import os
import os.path as osp
import numpy as np
from plyfile import PlyData
import ctypes as ct
import cv2
import time
import tqdm
import mmcv
from argparse import ArgumentParser

from utils.fat_utils import Scene, Side
from utils.class_names import fat_classes
from utils import coor_utils
from misc.uv_map import generate_uv_frag_map


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--txt-file',
        help='e.g. /<path_to_meta_data>/single_objects.txt')
    parser.add_argument(
        '--save-dir', default='data/fat/uvmap')
    parser.add_argument(
        '--data-root', default='data/fat')
    parser.add_argument(
        '--model-root', default='data/fat/ycb_models_nvdu_aligned_cm/')

    parser.add_argument(
        '--method', default='s')
    parser.add_argument(
        '--uvbins', type=int, default=256)
    parser.add_argument(
        '--fragbins', type=int, default=8)

    parser.add_argument(
        '--DEBUG', action="store_true",
        default=False,
        help="To show the generated images or not."
    )
    parser.add_argument(
        '--vis', action="store_true",
        default=True,
        help="visulaize generated images."
    )
    args = parser.parse_args()
    return args


def show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    _disp = cv2.resize(img.astype("uint8"), (480, 270))
    cv2.imshow(name, _disp)


class FATRenderDB:
    def __init__(self,
                 txt_file,
                 save_dir,
                 data_root,
                 model_root,
                 uv_method='s',
                 uv_bins=256,
                 frag_bins=8,
                 vis=False):
        with open(txt_file) as fp:
            lines = fp.readlines()
            lines = [v.strip().split(' ') for v in lines]

        self.lines = lines
        self.data_root = data_root
        self.save_dir = save_dir
        self.uv_method = uv_method
        self.uv_bins = uv_bins
        self.frag_bins = frag_bins

        self.vis = vis
        # Timing
        self.trans_time = []
        self.render_time = []

        so_p = 'third_party/raster_triangle/src/raster_nearest.so'
        self.dll = np.ctypeslib.load_library(so_p, '.')

        self.ply_dict = dict()
        _t = time.process_time()
        for model in fat_classes:
            # ply_file = os.path.join(model_root, model, 'google_512k/nontextured.ply')
            ply_file = os.path.join(model_root, model, 'google_16k/nontextured.ply')
            npts, xyz, r, g, b, n_face, face = self.load_ply_and_gen_uvfmap(ply_file)
            face = np.require(face, 'int32', 'C')
            r = np.require(r, 'float32', 'C')
            g = np.require(g, 'float32', 'C')
            b = np.require(b, 'float32', 'C')
            model_dict = dict(
                npts=npts,
                xyz=xyz,
                r=r,
                g=g,
                b=b,
                n_face=n_face,
                face=face)
            self.ply_dict[model] = model_dict
        _et = time.process_time()
        print(f"Finished loading PLYs, takes: {_et - _t}")

    def load_ply_and_gen_uvfmap(self, model_path):
        ply = PlyData.read(model_path)
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        pts = np.column_stack([x, y, z])
        uvf = generate_uv_frag_map(
            pts, method=self.uv_method, uv_bins=self.uv_bins,
            frag_bins=self.frag_bins, to_256=True)
        u_map, v_map, f_map = np.split(uvf, 3, axis=1)
        r = u_map
        g = v_map
        b = f_map
        face_raw = ply.elements[1].data
        face = []
        for item in face_raw:
            face.append(item[0])

        n_face = len(face)
        face = np.array(face).flatten()

        n_pts = len(x)
        xyz = np.stack([x, y, z], axis=-1)

        return n_pts, xyz, r, g, b, n_face, face

    def gen_pack_zbuf_render(self):
        for line in tqdm.tqdm(self.lines):
            pattern, num_str = line
            base = osp.join(self.data_root, pattern)
            num = int(num_str)
            scene = Scene(base, num)
            left = scene.left
            right = scene.right

            bgr = self._gen_pack_zbuf_render_side(left)
            fname = osp.join(self.save_dir, f"{pattern}/{num_str}.left.png")
            if bgr is not None:
                mmcv.imwrite(bgr, fname)
            else:
                print(f"{fname} is None.")

            bgr = self._gen_pack_zbuf_render_side(right)
            fname = osp.join(self.save_dir, f"{pattern}/{num_str}.right.png")
            if bgr is not None:
                mmcv.imwrite(bgr, fname)
            else:
                print(f"{fname} is None.")

            # print(f"Writing {fname}")

    def _gen_pack_zbuf_render_side(self,
                                   side: Side):
        h, w = side.cap_height, side.cap_width
        K = side.intrinsic_matrix[:3, :3]

        zbuf = np.require(np.ones(h*w)*1e9, 'float32', 'C')
        rbuf = np.require(np.zeros(h*w), 'int32', 'C')
        gbuf = np.require(np.zeros(h*w), 'int32', 'C')
        bbuf = np.require(np.zeros(h*w), 'int32', 'C')

        bgr = None

        for i, obj in enumerate(side.objects):
            Tmw = side.Tmw_list[i]
            R, T = Tmw[:3, :3], Tmw[:3, -1]
            cls_name = obj['class'].replace('_16k', '').replace('_16K', '')

            _t1 = time.process_time()
            new_xyz = self.ply_dict[cls_name]['xyz']  # .copy()
            new_xyz = np.dot(new_xyz, R.T) + T
            # p2ds = np.dot(new_xyz.copy(), K.T)
            p2ds = np.dot(new_xyz, K.T)
            # p2ds = p2ds[:, :2] / p2ds[:, 2:]
            p2ds = p2ds / p2ds[:, -1][:, None]
            p2ds = p2ds @ side.offset_matrix.T
            p2ds = coor_utils.from_home_nx(p2ds)
            _t2 = time.process_time()

            p2ds = np.require(p2ds.flatten(), 'float32', 'C')
            zs = np.require(new_xyz[:, 2].copy(), 'float32', 'C')

            self.dll.rgbzbuffer_noinit(
                ct.c_int(h),
                ct.c_int(w),
                p2ds.ctypes.data_as(ct.c_void_p),
                new_xyz.ctypes.data_as(ct.c_void_p),
                zs.ctypes.data_as(ct.c_void_p),
                self.ply_dict[cls_name]['r'].ctypes.data_as(ct.c_void_p),
                self.ply_dict[cls_name]['g'].ctypes.data_as(ct.c_void_p),
                self.ply_dict[cls_name]['b'].ctypes.data_as(ct.c_void_p),
                ct.c_int(self.ply_dict[cls_name]['n_face']),
                self.ply_dict[cls_name]['face'].ctypes.data_as(ct.c_void_p),
                zbuf.ctypes.data_as(ct.c_void_p),
                rbuf.ctypes.data_as(ct.c_void_p),
                gbuf.ctypes.data_as(ct.c_void_p),
                bbuf.ctypes.data_as(ct.c_void_p),
            )
            _t3 = time.process_time()

            zbuf.resize((h, w))
            msk = (zbuf > 1e-8).astype('uint8')
            if len(np.where(msk.flatten() > 0)[0]) < 500:
                continue
            zbuf *= msk.astype(zbuf.dtype) # * 1000.0

            bbuf.resize((h, w)), rbuf.resize((h, w)), gbuf.resize((h, w))
            bgr = np.concatenate((bbuf[:,:,None], gbuf[:, :, None], rbuf[:, :, None]), axis=2)
            bgr = bgr.astype('uint8')

            self.trans_time.append(_t2 - _t1)
            self.render_time.append(_t3 - _t2)

            # if self.vis:
                # show("bgr", bgr)
                # cv2.imshow("bgr", bgr.astype("uint8"))
            #     show_zbuf = zbuf.copy()
            #     min_d, max_d = show_zbuf[show_zbuf > 0].min(), show_zbuf.max()
            #     show_zbuf[show_zbuf>0] = (show_zbuf[show_zbuf>0] - min_d) / (max_d - min_d) * 255
            #     show_zbuf = show_zbuf.astype(np.uint8)
            #     cv2.imshow("dpt", show_zbuf)
            #     show_msk = (msk / msk.max() * 255).astype("uint8")
            #     cv2.imshow("msk", show_msk)
            #     cv2.waitKey(0)
            #
            # if self.DEBUG:
            #     cv2.imshow("rgb", rgb[:, :, ::-1].astype("uint8"))
            #     cv2.imshow("depth", (zbuf / zbuf.max() * 255).astype("uint8"))
            #     cv2.imshow("mask", (msk/ msk.max() * 255).astype("uint8"))
            #     cv2.waitKey(0)
        return bgr


def main():
    args = parse_args()
    gen = FATRenderDB(
        txt_file=args.txt_file,
        save_dir=args.save_dir,
        data_root=args.data_root,
        model_root=args.model_root,
        vis=args.vis,
        uv_method=args.method,
        uv_bins=args.uvbins,
        frag_bins=args.fragbins)
    gen.gen_pack_zbuf_render()
    print(f"Tranform time: {np.mean(gen.trans_time)}")  # usually 0.0002
    print(f"Render time: {np.mean(gen.render_time)}")  # usually 0.01


if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
