import numpy as np
import cv2
import json
import time
import os
import argparse
import rpyc
from estimater import *
from datareader import *
import argparse
import logging

from rpyc.utils.server import ThreadedServer

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
rpyc.core.protocol.DEFAULT_CONFIG['allow_public_attrs'] = True

logger = logging.getLogger(__name__)

@rpyc.service
class FoundationPoseService(rpyc.Service):
    @rpyc.exposed
    def setup(self, log_level=logging.INFO, log_file="FoundationPoseService.log", debug_level = 1 ):
        '''
        setup service 
        
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s : %(name)s  : %(funcName)s : %(levelname)s : %(message)s", 
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )
        '''
        logger.setLevel(log_level)
        fileHandler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s : %(name)s  : %(funcName)s : %(levelname)s : %(message)s")
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(formatter)
        logger.addHandler(consoleHandler)

        if hasattr(logger, "debug_level"):
            logger.debug_level = debug_level
        else:
            setattr(logger, "debug_level", debug_level)
        print(f"log_file: {logger.handlers[0].baseFilename}")


    @rpyc.exposed
    class VTFoundationPose(object):
        '''
        CNOS vision tool: detect objects given the CAD model at test time. 
        '''
        @rpyc.exposed
        def __init__(self, mesh_file,mesh_scale = 0.001, est_refine_iter = 5, track_refine_iter = 2):
            '''
            initalize tool by creating the tool model.
            '''
            self.log_dir = os.path.dirname(logger.handlers[0].baseFilename)

            trimesh.units.unit_conversion('millimeters', 'meters')
            mesh = trimesh.load(mesh_file)
            mesh.apply_scale(mesh_scale)

            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

            scorer = ScorePredictor()
            refiner = PoseRefinePredictor()
            glctx = dr.RasterizeCudaContext()
            est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, \
                                scorer=scorer, refiner=refiner, debug_dir=self.log_dir, \
                                debug=logger.debug_level, glctx=glctx)
            
            logger.info("Model initialization done")

            self.tool_model = {'model': est, 'mesh': mesh, 'to_origin' : to_origin, 'bbox' : bbox}
            self.tool_setting = {'est_refine_iter' : est_refine_iter,
                            'track_refine_iter' : track_refine_iter
            }
            self.tool_input = {}
            self.tool_result = None

        @rpyc.exposed
        def __call__(self, rgb, depth, K, masks=None, **kwargs):
            return self.predicate_many(rgb, depth, K, masks, **kwargs)
    
        @rpyc.exposed
        def track_one(self, rgb, depth, K, mask=None, **kwargs):
            '''
            do the actual processing by executing the tool model

            return: tool_result
            '''
            # get model 
            est = self.tool_model['model']
            est_refine_iter = self.tool_setting['est_refine_iter']
            track_refine_iter = self.tool_setting['track_refine_iter']

            # run inferenced
            if mask is not None:
                mask = mask.astype(bool)
                pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=est_refine_iter)
                self.tool_input.update({'rgb': rgb, 'depth': depth, 'mask': mask, 'K':K})
            else:
                pose = est.track_one(rgb=rgb, depth=depth, K=K, iteration=track_refine_iter)
                self.tool_input.update({'rgb': rgb, 'depth': depth, 'K':K})                      
            pose.reshape(4,4)

            # save internal state for post processing or visualization
            self.tool_result = {'pose': pose}

            return pose

        @rpyc.exposed
        def predict_many(self, rgb, depth, K, masks=None, **kwargs):
            '''
            detect multiple instances of a mesh model. The detection masks are given by masks as a list.

            return: tool_result
            '''
            # get model 
            est = self.tool_model['model']
            est_refine_iter = self.tool_setting['est_refine_iter']
            track_refine_iter = self.tool_setting['track_refine_iter']

            # get local copy of rgb, depth
            rgb_local = np.array(rgb)
            depth_local = np.array(depth)
            K_local = np.array(K)

            # run inference
            poses = []
            masks_local = []
            for i, mask in enumerate(masks):
                maskb = np.array(mask).astype(dtype=np.bool_)
                masks_local.append(maskb)
                pose = est.register(K=K_local, rgb=rgb_local, depth=depth_local, ob_mask=maskb, iteration=est_refine_iter)
                poses.append(pose.reshape(4,4))

            # save internal state for post processing or visualization
            self.tool_input.update({'rgb': rgb_local, 'depth': depth_local, 'masks': masks_local, 'K':K_local})
            self.tool_result = {'poses': poses}

            return poses


        @rpyc.exposed
        def visualize(self, bg):
            '''
            pose processing to generate visualization
            '''
            K = self.tool_input['K']
            vis = np.array(bg)

            if 'pose' in self.tool_result:
                pose = self.tool_result['pose']
                center_pose = pose@np.linalg.inv(self.tool_model['to_origin'])
                vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=self.tool_model['bbox'])
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
            elif 'poses' in self.tool_result:
                for pose in self.tool_result['poses']:                
                    center_pose = pose@np.linalg.inv(self.tool_model['to_origin'])
                    vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=self.tool_model['bbox'])
                    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
            self.tool_result['vis'] = vis
            return vis

        @rpyc.exposed
        def log(self, context):
            '''
            save results
            '''
            # save poses into a text file
            if 'pose' in self.tool_result:
                np.savetxt(f'{self.log_dir}/{context}.pose.txt', self.tool_result['pose'])
            elif 'poses' in self.tool_result:
                np.savetxt(f'{self.log_dir}/{context}.poses.txt', self.tool_result['poses'])
            cv2.imwrite(f'{self.log_dir}/{context}.color.png', self.tool_input['rgb'])
            cv2.imwrite(f'{self.log_dir}/{context}.depth.png', self.tool_input['depth'])
            if 'vis' in self.tool_result:
                cv2.imwrite(f'{self.log_dir}/{context}.vis.png', self.tool_result['vis'])
            return
        

if __name__ == "__main__":
    server = ThreadedServer(FoundationPoseService, port = 12346, protocol_config = rpyc.core.protocol.DEFAULT_CONFIG)
    server.start()        