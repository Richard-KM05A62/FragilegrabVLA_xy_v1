from pyAgxArm import create_agx_arm_config, AgxArmFactory

cfg = create_agx_arm_config(
    robot="piper_x",
    comm="can",
    channel="0",
    interface="agx_cando",
)
robot = AgxArmFactory.create_arm(cfg)
robot.connect()
print("CAN 连接状态:", robot.is_connected())
print("CAN 通信正常:", robot.is_ok())
