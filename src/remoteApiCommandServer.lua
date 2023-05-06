function __setObjectPosition__(a,b,c)
    -- compatibility routine, wrong results could be returned in some situations, in CoppeliaSim <4.0.1
    if b==sim.handle_parent then
        b=sim.getObjectParent(a)
    end
    if (b~=-1) and (sim.getObjectType(b)==sim.object_joint_type) and (sim.getInt32Parameter(sim.intparam_program_version)>=40001) then
        a=a+sim.handleflag_reljointbaseframe
    end
    return sim.setObjectPosition(a,b,c)
end
function __setObjectOrientation__(a,b,c)
    -- compatibility routine, wrong results could be returned in some situations, in CoppeliaSim <4.0.1
    if b==sim.handle_parent then
        b=sim.getObjectParent(a)
    end
    if (b~=-1) and (sim.getObjectType(b)==sim.object_joint_type) and (sim.getInt32Parameter(sim.intparam_program_version)>=40001) then
        a=a+sim.handleflag_reljointbaseframe
    end
    return sim.setObjectOrientation(a,b,c)
end

importShape = function(inInts, inFloats, inStrings, inBuffer)
    -- inInts: size = 4, not used here
    -- inFloats: size = 9, respectively represent position, orientation and code for color
    -- inStrings: [curr_mesh_file, curr_shape_name]
    -- inBuffer: bytearray()

    inMeshPath = inStrings[1]
    inShapeName = inStrings[2]
    inShapePosition = {inFloats[1], inFloats[2], inFloats[3]}
    inShapeOrientation = {inFloats[4], inFloats[5], inFloats[6]}
    inShapeShapeColor = {inFloats[7], inFloats[8], inFloats[9]}
    robotHandle = sim.getObjectHandle('base_link')
    shapeHandle = sim.importShape(0, inMeshPath, 0, 0, 1)
    sim.setObjectName(shapeHandle, inShapeName)
    __setObjectPosition__(shapeHandle, robotHandle, inShapePosition)
    __setObjectOrientation__(shapeHandle, robotHandle, inShapeOrientation)

    sim.setShapeColor(shapeHandle, nil, sim.colorcomponent_ambient_diffuse, inShapeShapeColor)

    sim.setObjectInt32Parameter(shapeHandle, sim.shapeintparam_static, 0)
    sim.setObjectInt32Parameter(shapeHandle, sim.shapeintparam_respondable, 1)

    -- cupHandle = sim.getObjectHandle('Cup')

    -- print('hello world!')
    -- mass, inertia, com = sim.getShapeMassAndInertia(cupHandle)
    -- mass, inertia, com = sim.getShapeMassAndInertia(shapeHandle)
    -- sim.setShapeMassAndInertia(shapeHandle, 0.5, inertia, com, nil)
    -- sim.setShapeMassAndInertia(shapeHandle, 100, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 0, 0}, nil)
    sim.resetDynamicObject(shapeHandle)
    -- sim.setModelProperty(shapeHandle, sim.modelproperty_not_dynamic)
    -- print(sim.getModelProperty(shapeHandle, sim.modelproperty_not_dynamic))
    return {shapeHandle}, {}, {}, ''
end

createPureshape = function(inInts, inFloats, inStrings, inBuffer)


    primitiveType = inInts[1]
    print(primitiveType)
    options = 8
    
    mass = 3.351e-02
    precision = nullptr
    
    -- prototype for sim.createPureshape
    -- primitiveType: 0 for a cuboid, 1 for a sphere, 2 for a cylinder and 3 for a cone

    -- options: Bit-coded: if bit0 is set (1), backfaces are culled. 
    -- If bit1 is set (2), edges are visible. 
    -- If bit2 is set (4), the shape appears smooth. 
    -- If bit3 is set (8), the shape is respondable.
    --  If bit4 is set (16), the shape is static. 
    --  If bit5 is set (32), the cylinder has open ends

    -- sizes: 3 values indicating the size of the shape

    -- mass: the mass of the shape

    -- precision: 2 values that allow specifying the number of sides and faces of a cylinder or sphere. Can be nullptr for default values


    robotHandle = sim.getObjectHandle('base_link_respondable')

    objHandle = sim.createPureshape(primitiveType, options, sizes, mass, precision)
    
    if (objHandle ~= -1) then

        inShapePosition = {inFloats[1], inFloats[2], inFloats[3]}
        inShapeOrientation = {inFloats[4], inFloats[5], inFloats[6]}
        inShapeShapeColor = {inFloats[7], inFloats[8], inFloats[9]}
        diameter = inFloats[10]
        sizes = {diameter, diameter, diameter}

        currShapeName = inStrings[1]

        sim.setObjectName(objHandle,currShapeName)
        sim.setShapeColor(objHandle, nil, sim.colorcomponent_ambient_diffuse, inShapeShapeColor)
        sim.resetDynamicObject(objHandle)
        __setObjectPosition__(objHandle, robotHandle, inShapePosition)
        __setObjectOrientation__(objHandle, robotHandle, inShapeOrientation)
        

    end
    return {objHandle}, {}, {}, ''
end


