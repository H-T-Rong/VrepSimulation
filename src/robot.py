import remote
import socket
import select
import struct
import time
import os
import numpy as np
import utils
from remote import sim

pi = np.pi

class Robot:
    def __init__(self, motorName='motor', obj_mesh_dir = './blocks', num_obj=0, num_primitive_obj=0) -> None:
        
        self.objectsDict = dict()
        self.motor = None

        sim.simxFinish(-1) # just in case, close all opened connections

        # 端口号默认要用19997, 在remoteApiConnections有定义
        self.clientID=sim.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to CoppeliaSim
        if self.clientID !=-1:
            print('Connected to remote API server')
        else:
            print('Failed connecting to remote API server')

        ret, self.motor = sim.simxGetObjectHandle(self.clientID,
                                             motorName,
                                             sim.simx_opmode_blocking)
        
        if ret == sim.simx_return_ok:
            print('successfully initiate the robot')
        else:
            print('motor name incorrect')

        self.obj_mesh_dir = obj_mesh_dir
        self.workspace_limits = None # 需要预实验确定小球的投放范围
        self.num_primitive_obj = 0

        self.color_space = np.asarray([[78.0, 121.0, 167.0], # blue
                                        [89.0, 161.0, 79.0], # green
                                        [156, 117, 95], # brown
                                        [242, 142, 43], # orange
                                        [237.0, 201.0, 72.0], # yellow
                                        [186, 176, 172], # gray
                                        [255.0, 87.0, 89.0], # red
                                        [176, 122, 161], # purple
                                        [118, 183, 178], # cyan
                                        [255, 157, 167]])/255.0 #pink


    def getObjectHandle(self, objectName):
        """get the handle of object according to its name 

        Args:
            objectName (string): object's name

        Returns:
            int/ None: the handle of the object, when there is no such name in vrep, return None 
        """
        clientID = self.clientID

        ret, targetObj = sim.simxGetObjectHandle(clientID, objectName, 
                        sim.simx_opmode_blocking)
        
        if ret == sim.simx_return_ok:
            if objectName not in self.objectsDict:
                self.objectsDict[objectName] = targetObj

            return targetObj
        else:
            print('fail to get Object Handle: '+ objectName)
            return None

    
    def addObjects(self, num_obj):
        # Define colors for object meshes (Tableau palette)


        # Read files in object mesh directory
        self.num_obj = num_obj
        self.mesh_list = os.listdir(self.obj_mesh_dir)

        # Randomly choose objects to add to scene
        self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
        self.obj_mesh_color = self.color_space[np.asarray(range(self.num_obj)) % 10, :]

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        self.object_handles = []
        sim_obj_handles = []
        for object_idx in range(len(self.obj_mesh_ind)):
            curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])

            curr_shape_name = 'shape_%02d' % object_idx
            drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
            drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            
            object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1], self.obj_mesh_color[object_idx][2]]

            # =================================Add Object=========================================
            ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = sim.simxCallScriptFunction(self.clientID, # vrep client id
                                                                                              'remoteApiCommandServer', # script description
                                                                                              sim.sim_scripttype_childscript, # scriptHandleOrType
                                                                                              'importShape', # functionName
                                                                                              [0,0,255,0], # inInts
                                                                                              object_position + object_orientation + object_color, # inFloats
                                                                                              [curr_mesh_file, curr_shape_name], # inStrings
                                                                                               bytearray(), # inBuffer
                                                                                               sim.simx_opmode_blocking) # vrep operation mode
            # =====================================================================================

            if ret_resp != sim.simx_return_ok:
                print('Failed to add new objects to simulation. Please restart.')
                exit()
            curr_shape_handle = ret_ints[0]
            self.object_handles.append(curr_shape_handle)

            time.sleep(2)


    def addSphereObject(self, diameter, mass=None):
        # sim.createpureshape
        # 小球投放的位置应该选用相对位置，否则会被以及存在的物体挤出去
        if 'motor' not in self.objectsDict:
            self.getObjectHandle('motor')
        if 'rotate_link' not in self.objectsDict:
            self.getObjectHandle('rotate_link_respondable')
        if 'bowl_link' not in self.objectsDict:
            self.getObjectHandle('bowl_link_visual')

        rotate_link_pos, rotate_link_ori = self.getPosition(self.objectsDict['rotate_link_respondable'])
        bowl_link_pos, bowl_link_ori = self.getPosition(self.objectsDict['bowl_link_visual'])

        ref, bowl_height = sim.simxGetObjectFloatParameter(self.clientID,
                                                      self.objectsDict['bowl_link_visual'],
                                                      parameterID=sim.sim_objfloatparam_objbbox_max_z,
                                                      operationMode=sim.simx_opmode_blocking)

        ref, bowl_min_x = sim.simxGetObjectFloatParameter(self.clientID,
                                                      self.objectsDict['bowl_link_visual'],
                                                      parameterID=sim.sim_objfloatparam_objbbox_min_x,
                                                      operationMode=sim.simx_opmode_blocking)
        
        ref, bowl_max_x = sim.simxGetObjectFloatParameter(self.clientID,
                                                      self.objectsDict['bowl_link_visual'],
                                                      parameterID=sim.sim_objfloatparam_objbbox_max_x,
                                                      operationMode=sim.simx_opmode_blocking)
        

        bowl_diameter = bowl_max_x - bowl_min_x

        drop_x = rotate_link_pos[0] + bowl_diameter/(4 ) * np.cos(rotate_link_ori[2])
        drop_y = rotate_link_pos[1] + bowl_diameter/(4 ) * np.sin(rotate_link_ori[2])
        drop_z = rotate_link_pos[2] + bowl_height - diameter/2 + 0.01

        # drop_z = rotate_link_max_z - diameter/2 -0.01
        object_position = [drop_x, drop_y, drop_z]
        curr_shape_name = 'sphere%d' % self.num_primitive_obj
        object_orientation = [2*np.pi*np.random.random_sample(), 
                              2*np.pi*np.random.random_sample(), 
                              2*np.pi*np.random.random_sample()]
        
        object_color = self.color_space[self.num_primitive_obj%8, :].tolist()
        ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = sim.simxCallScriptFunction(self.clientID, # vrep client id
                                                                                'remoteApiCommandServer', # script description
                                                                                sim.sim_scripttype_childscript, # scriptHandleOrType
                                                                                'createPureshape', # functionName
                                                                                [1], # inInts: Sphere
                                                                                object_position + object_orientation + object_color + [diameter], # inFloats
                                                                                [curr_shape_name], # inStrings
                                                                                bytearray(), # inBuffer
                                                                                sim.simx_opmode_blocking) # vrep operation mode

        self.objectsDict[curr_shape_name] = ret_ints[0]
        ret = self.setPosition(ret_ints[0], object_position)
        if mass is not None:
            sim.simxSetObjectFloatParameter(clientID=self.clientID,
                                            objectHandle=ret_ints[0],
                                            parameterID=sim.sim_shapefloatparam_mass,
                                            parameterValue=mass,
                                            operationMode=sim.simx_opmode_blocking)

    def request_dataStreaming(self, request_types, targetObj):

        request_successfull = True
        for idx,request_type in enumerate(request_types):
            if request_type == 'Force':
                ret, force = sim.simxGetJointForce(self.clientID,
                                               targetObj[idx],
                                               operationMode=sim.simx_opmode_streaming)
            if request_type == 'Velocity':
                ret, linear, angular = sim.simxGetObjectVelocity(self.clientID,
                                    targetObj[idx],
                                    sim.simx_opmode_streaming)
            
            if ret == sim.simx_return_novalue_flag:
                print('request for ' + request_type + ' is successfull')
            else:
                print('request for ' + request_type + ' is unsuccessfull')
                request_successfull = False
        
        return request_successfull

    def getForce(self, targetObj, opmode='blocking'):
        
        if targetObj != self.motor:
            raise RuntimeError('not advisable getForce object')
        
        if opmode == 'streaming':
            ret, force = sim.simxGetJointForce(self.clientID,
                                               targetObj,
                                               operationMode=sim.simx_opmode_buffer)
        else:
            ret, force = sim.simxGetJointForce(self.clientID,
                                               targetObj,
                                               sim.simx_opmode_blocking)
        
        if ret == sim.simx_return_ok:
            return force
        elif ret == sim.simx_return_novalue_flag:
            print('waiting for remote server to start data streaming service')
            return None
        else:
            print('fail to get force')
            # self.exitSimulation()
            return None

    def getVelocity(self, targetObj, opmode='blocking'):
        
        if opmode == 'streaming':
            ret, linear, angular = sim.simxGetObjectVelocity(self.clientID,
                                    targetObj,
                                    sim.simx_opmode_buffer)
            
        else:
            ret, linear, angular = sim.simxGetObjectVelocity(self.clientID,
                                                             targetObj,
                                                             sim.simx_opmode_blocking)
        if ret == sim.simx_return_ok:
            return (linear, angular)
        elif ret == sim.simx_return_novalue_flag:
            print('waiting for remote server to start data streaming service')
            return (None, None)
        else:
            print('fail to get Velocity')

            return (None, None)
        
         

    def setVelocity(self, targetVelocity, opmode='blocking'):
        # if targetVelocity > 180:
        #     targetVelocity = 180
        warnings = False
        if targetVelocity > 2000 * pi:
            print('robot.py::setVelocity(): Warning! too fast target velocity')
            warnings = True
            # raise ValueError('not current target velocity, please convert the angle from deg to rad')
        # targetVelocity = targetVelocity * np.pi / 180
        # 将角度制转换为弧度制

        if opmode == 'non-blocking':
            ret = sim.simxSetJointTargetVelocity(
                self.clientID,
                self.motor,
                targetVelocity=targetVelocity,
                operationMode=sim.simx_opmode_oneshot
            )
            self.last_target_velocity = targetVelocity
        else: 
            ret = sim.simxSetJointTargetVelocity(
                self.clientID,
                self.motor,
                targetVelocity=targetVelocity,
                operationMode=sim.simx_opmode_blocking
            )
            try:
                if ret != sim.simx_return_ok:
                    print('fail to set velocity')
                    # self.exitSimulation()
                else:
                    # print('successfully set velocity')
                    pass
            except TypeError:
                pass
            
            self.last_target_velocity = targetVelocity

            return warnings
        

        

    def getPosition(self, targetObj):

        clientID = self.clientID

        ret_coor, coor = sim.simxGetObjectPosition(clientID, 
                                             targetObj, 
                                             -1, 
                                             sim.simx_opmode_blocking)
        
        ret_ori, ori = sim.simxGetObjectOrientation(clientID,
                                                targetObj,
                                                -1,
                                                sim.simx_opmode_blocking)
        
        
        if ret_coor == sim.simx_return_ok and ret_ori == sim.simx_return_ok:
            return (coor, ori)
        else:
            print('fail to get position of object {}'.format(targetObj))
            return None
        
    def setPosition(self, targetObj, coor):

        if type(coor) != list:
            try:
                coor = list(coor)
            except:
                coor = coor.tolist()
            
        if len(coor) != 3:
            print('Invalid coordinate, coordinate form is (x,y,z)')
            # fail to set position
            return False
        

        clientID = self.clientID
        ret = sim.simxSetObjectPosition(clientID, targetObj,-1,(coor[0],coor[1],coor[2]), 
                          sim.simx_opmode_blocking)
        if ret == sim.simx_return_ok:
            return True
        else:
            print('fail to set object Position')
            return False


    def checkConnection(self):
        # Retrieves the time needed for a command to be sent to the server, executed, 
        #     and sent back. That time depends on various factors like the client settings,
        #     the network load, whether a simulation is running, whether the simulation
        #     is real-time, the simulation time step, etc. The function is blocking. 

        clientID = self.clientID
        ret, pingTime = sim.simxGetPingTime(clientID)

        print('pingTime: ', pingTime)
        
        return pingTime
    
    def stopSimulation(self):
        clientID = self.clientID
        ret = sim.simxStopSimulation(clientID=clientID, operationMode=sim.simx_opmode_blocking)
        print('Stop the simulation')
        

    def pauseSimulation(self):
        clientID = self.clientID
        # ret = sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
        ret = sim.simxPauseSimulation(clientID=clientID, operationMode=sim.simx_opmode_blocking)

        print('pause the simulation')
        time.sleep(0.5)#需要反应时间

        
    
    def resumeSimulation(self):
        clientID = self.clientID
        ret = sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
        print('resume the simulation')

        time.sleep(0.5)


    def exitSimulation(self):
        clientID = self.clientID
        sim.simxGetPingTime(clientID)
        
        sim.simxFinish(-1) # just in case, close all opened connections
        ret = sim.simxPauseSimulation(clientID=clientID, operationMode=sim.simx_opmode_blocking)
        print('pause simulation')

def printStatus(robot):
    print('velocity: ')
    print(robot.getVelocity(rotate_link))

    print('force: ')
    print(robot.getForce(motor_link))

def floatlist2str(x):
    return '{:.6f}'.format(x)

if __name__ == "__main__":
    output_dir = './output'
    

    num_pos_data = 1000
    
    count = 0

    draw_trajactory = False
    test_generateObj = True
    test_API = False

    robot = Robot()
    
    # do Something

    rotate_link = robot.getObjectHandle('rotate_link_respondable')
    motor_link = robot.motor
    sphere_link = robot.getObjectHandle('Sphere')


    if draw_trajactory:
        try:
            robot.setVelocity(90)
            coors_record = []
            oris_record = []
            time_recorder = time.perf_counter()
            
            while(count < num_pos_data):
                if time.perf_counter() - time_recorder > 0.1:

                    curr_coor, curr_ori = robot.getPosition(sphere_link)
                    coors_record.append(','.join(['{:6f}'.format(coor) for coor in curr_coor]))
                    oris_record.append(','.join(['{:6f}'.format(ori) for ori in curr_ori]))
                
                    count += 1

                    time_recorder = time.perf_counter()
            
            
            printStatus(robot=robot)

        except:

            robot.exitSimulation() # just in case, close all opened connections
            raise RuntimeError('Get Error, needed to be handled')

        with open('./output/workspace_record', 'w+') as f:
            f.write('\n'.join(coors_record))


    if test_generateObj:
        robot.setVelocity(0)
        print('adding sphere')
        robot.addSphereObject(diameter=8e-2, mass=None)
        print('done! ')
        # robot.setVelocity(90)
        time.sleep(10)
        
        robot.stopSimulation()
        robot.exitSimulation()



    if test_API:
        robot.setVelocity(0)
        robot.stopSimulation()
        time.sleep(1)
        robot.resumeSimulation()
        

        # robot.getVelocity(robot.motor)
        # robot.stopSimulation()
        # time.sleep(10)
        # print('simulation stopped')
        # print('everything back to original state')
        # robot.resumeSimulation()
        # print('simulation start again')
        # time.sleep(5)
        









# for reference:

# A remote API function return can be a combination of following flags:

# simx_return_ok (0)
# The function executed fine
# simx_return_novalue_flag (1 (i.e. bit 0))
# There is no command reply in the input buffer. This should not always be considered as an error, depending on the selected operation mode
# simx_return_timeout_flag (2 (i.e. bit 1))
# The function timed out (probably the network is down or too slow)
# simx_return_illegal_opmode_flag (4 (i.e. bit 2))
# The specified operation mode is not supported for the given function
# simx_return_remote_error_flag (8 (i.e. bit 3))
# The function caused an error on the server side (e.g. an invalid handle was specified)
# simx_return_split_progress_flag (16 (i.e. bit 4))
# The communication thread is still processing previous split command of the same type
# simx_return_local_error_flag (32 (i.e. bit 5))
# The function caused an error on the client side
# simx_return_initialize_error_flag (64 (i.e. bit 6))
# simxStart was not yet called