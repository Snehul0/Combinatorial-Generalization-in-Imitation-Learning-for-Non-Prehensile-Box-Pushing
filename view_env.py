import mujoco
import mujoco.viewer
import time

XML_PATH = "scene.xml"   # use your actual environment file

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Environment loaded. Close the viewer window to exit.")

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
