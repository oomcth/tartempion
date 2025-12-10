import os, time, warnings
from contextlib import nullcontext

import numpy as np
import pinocchio as pin
from coal import AABB
from candlewick import Visualizer, VisualizerConfig

from diffusion_gps.utils import colors
from diffusion_gps.visualization.render.screenshots import video_to_timeline
from diffusion_gps.visualization.render.base import RendererAbstract
from diffusion_gps.tasks.base import TaskInstance


class RendererCandlewick(RendererAbstract):
    viz: Visualizer = None
    default_camera_position = np.array([0.0, 1.2, 0.7])

    def wait_until_ended_if_needed(self):
        if self.viz is not None and not self.disable:
            while not self.viz.shouldExit:
                self.display()

    def init_visualizer(self):
        config = VisualizerConfig()
        config.width = 1280
        config.height = 720
        self.viz = viz = Visualizer(config, self.task.rmodel, self.task.gmodel)
        viz.setCameraPosition(self.default_camera_position)
        viz.setCameraTarget(np.zeros(3))
        # viz.worldSceneBounds = AABB(np.array([-10.0,-10,-10]),np.array([10.,10,10]))

    def rebuild(self):
        if self.viz is not None and self.must_rebuild:
            del self.viz
            self.init_visualizer()
        self.must_rebuild = False

    def add_object(self, geom_obj: pin.GeometryObject):
        geom_obj.overrideMaterial = True
        super().add_object(geom_obj)

    def render_traj(
        self,
        task_instance: TaskInstance,
        xtraj: np.ndarray,
        cmap: colors.Colormap = colors.get_cst_cmap(colors.COLOR_TRAIL),
        method="Policy",
    ):
        try:
            if self.viz is not None and self.viz.shouldExit:
                self.disable = True
            if self.disable:
                return ()
            if self.record:
                os.makedirs(self.path_videos, exist_ok=True)
            self.setup_instance(task_instance)
            self.nb_trajs_rendered += 1
            xtraj = np.array(xtraj)
            xtraj, fps = self.sample_frames(xtraj)
            n_frames = len(xtraj)
            qtraj = xtraj[:, : self.task.nq]
            vtraj = xtraj[:, self.task.nq :]
            if self.record:
                warnings.warn("Renderer.record using Candlewick not implemented yet")
            eef_traj = []

            for i, (q, v) in enumerate(zip(qtraj, vtraj)):
                pin.forwardKinematics(self.task.rmodel, self.viz.data, q, v)
                if self.draw_eef_traj:
                    ee_pos = pin.updateFramePlacement(
                        self.task.rmodel, self.viz.data, task_instance.frame_id_target
                    )
                    trail_obj = self.get_object(self.trail_objects_names[i])
                    # color = cmap(i / n_frames) # TODO DEBUG trail...
                    color = cmap(0.0)
                    color = list(color)
                    color[-1] = 1.0  # TODO DEBUG alpha brightness
                    color = tuple(color)
                    self.change_object(trail_obj, meshColor=color)
                    trail_obj.placement = ee_pos
                    eef_traj.append((ee_pos, color))
                if self.viz.shouldExit:
                    self.disable = True
                    break
                self.display(q)
                time.sleep(2 * 1 / fps)
            if self.draw_eef_traj:
                SE3_start = self.task.forward_kinematics(
                    task_instance.q_init, task_instance.frame_id_target
                )
                self.eef_trajs.append(
                    (
                        SE3_start,
                        task_instance.frame_SE3_target,
                        eef_traj,
                    )
                )

        except KeyboardInterrupt:
            print("Candlewick terminated using KeyboardInterrupt")
            self.disable = True
        if self.disable:
            del self.viz
