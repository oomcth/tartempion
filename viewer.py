from pinocchio.visualize import MeshcatVisualizer as Visualizer
import pinocchio as pin
import numpy as np
import matplotlib.pyplot as plt
import meshcat.geometry as g


def view(model, collision_model, visual_model):
    viz = Visualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel(color=[0.0, 0.0, 0.0, 1.0])
    viz.displayCollisions(True)
    viz.display(pin.neutral(model))


class Viewer:
    def __init__(self, model, collision_model, visual_model, open=True):
        self.viz = Visualizer(model, collision_model, visual_model)
        self.viz.initViewer(open=open)
        self.viz.loadViewerModel(color=[0.5, 0.5, 0.5, 0.5])
        self.viz.setBackgroundColor()

    def display(self, q, frame_id=257):
        self.viz.displayFrames(True, [frame_id], 0.2, 5)
        self.viz.display(q)

    def screenshot(
        self,
        model: pin.Model,
        q,
        pos=0,
        savedir="screenshots/",
        filename="",
        res=(100, 100),
        plot=False,
    ):
        pin.forwardKinematics(model, model.data, q)
        pin.updateFramePlacements(model, model.data)
        if pos == 0 | pos == 1 or pos == 2:
            if pos == 0:
                id = model.getFrameId("universe")
            elif pos == 1:
                id = model.getFrameId("left_camera_infra2_frame")
            elif pos == 2:
                id = model.getFrameId("left_camera_infra2_frame")
            self.viz.setCameraPosition(model.data.oMf[id].act(np.array([0.01, 0.0, 0])))
            self.viz.setCameraTarget(model.data.oMf[id].act(np.array([0.3, 0.0, 0])))
        self.viz.display(q)
        img: np.ndarray = self.viz.captureImage(*res)
        if filename != "":
            np.save(savedir + filename, img)
        if plot:
            plt.imshow(img)
            plt.axis("off")
            plt.show()
        return img

    def play(
        self, qs, dt, model: pin.Model, vs, screen=False, show=False, set_hand_cam=False
    ):
        sphere_name2 = "toolplay"
        self.viz.viewer[sphere_name2].set_object(
            g.Sphere(0.052), g.MeshLambertMaterial(color=0x00FF00)
        )

        def callback(i: int):
            pin.framesForwardKinematics(model, model.data, qs[i])
            pin.updateFramePlacements(model, model.data)
            M_tool = model.data.oMf[15]
            self.viz.viewer[sphere_name2].set_transform(M_tool.homogeneous)

            if set_hand_cam:
                id = model.getFrameId("left_camera_infra2_frame")
                self.viz.setCameraPosition(
                    model.data.oMf[id].act(np.array([0.01, 0.0, 0]))
                )
                self.viz.setCameraTarget(
                    model.data.oMf[id].act(np.array([0.3, 0.0, 0]))
                )
            if screen:
                img = self.viz.captureImage(100, 100)
                if show:
                    plt.imshow(img)
                    plt.axis("off")
                    plt.show()

        self.viz.play(qs, dt, callback)
        self.viz.viewer.delete()
