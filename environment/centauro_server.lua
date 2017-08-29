package.path=package.path .. ";/home/xi/workspace/centauro_planer/environment/?.lua"
require("common_functions")
require("ompl_functions")
require("robot_control")

-- simSetThreadSwitchTiming(2) 
-- simExtRemoteApiStart(19999)
function reset(inInts,inFloats,inStrings,inBuffer)
    local radius = inFloats[1]
    init(radius)
    return {}, {}, {}, ''
end

function step(inInts,inFloats,inStrings,inBuffer)
    -- print('step')
    res = do_action_rl(_robot_hd, inFloats)
    -- sample_obstacle_position(obs_hds, #obs_hds)

    return {}, {}, {}, res
end

function move_robot(inInts,inFloats,inStrings,inBuffer)
    -- print('step')
    local robot_pos = simGetObjectPosition(_robot_hd, -1)
    robot_pos[1] =  inFloats[1]
    robot_pos[2] =  inFloats[2]

    simSetObjectPosition(_robot_hd, -1, robot_pos)
    -- sample_obstacle_position(obs_hds, #obs_hds)
    return {}, {}, {}, ''
end

function get_obstacle_info(inInts,inFloats,inStrings,inBuffer)
    local obstacle_dynamic_collection = simGetCollectionHandle('obstacle_dynamic')
    local obstacle_dynamic_hds = simGetCollectionObjects(obstacle_dynamic_collection)

    local obs_info = {}
    for i=1, #obstacle_dynamic_hds, 1 do 
        local pos = simGetObjectPosition(obstacle_dynamic_hds[i], -1)
        local res, type, dim = simGetShapeGeomInfo(obstacle_dynamic_hds[i])
        obs_info[#obs_info+1] = pos[1]
        obs_info[#obs_info+1] = pos[2]

        obs_info[#obs_info+1] = dim[1]
        obs_info[#obs_info+1] = dim[2]
        obs_info[#obs_info+1] = dim[3]

        print('shape: ', dim[1], dim[2], dim[3], dim[4])
    end

    return {}, obs_info, {}, ''
end

function get_robot_state(inInts,inFloats,inStrings,inBuffer)
    local target_pos =simGetObjectPosition(_target_hd, _robot_hd)
    local target_ori =simGetObjectPosition(_target_hd, _robot_hd)

    local pos =simGetObjectPosition(_robot_hd,-1)
    local ori =simGetObjectQuaternion(_robot_hd,-1)
    -- local joint_pose = get_joint_values(_joint_hds)
    local leg_l = get_current_l(_robot_hd)

    -- x, y, theta, h, l,   tx, ty, t_theta,   t_h, t_l
    local state = {}
    state[1] = pos[1]
    state[2] = pos[2]
    state[3] = ori[3]
    state[4] = pos[3]
    state[5] = leg_l

    state[6] = target_pos[1]
    state[7] = target_pos[2]
    state[8] = target_ori[3]

    state[9] = 0.1
    state[10] = 0.0

    local obstacle_dynamic_collection = simGetCollectionHandle('obstacle_dynamic')
    local obstacle_dynamic_hds = simGetCollectionObjects(obstacle_dynamic_collection)
    for i=1, #obstacle_dynamic_hds, 1 do 
        local obs_pos = simGetObjectPosition(obstacle_dynamic_hds[i], _robot_hd)
        local obs_pos_global = simGetObjectPosition(obstacle_dynamic_hds[i], -1)
        -- local res, type, dim = simGetShapeGeomInfo(obstacle_dynamic_hds[i])
        
        local x = math.abs(obs_pos_global[1])
        local y = math.abs(obs_pos_global[2])

        if x < 2.5 and y < 2.5 then 
            state[#state+1] = obs_pos[1]
            state[#state+1] = obs_pos[2] 
        end
    end

    -- print ('in get robot state:', #state[3])
    return {}, state, {}, ''
end

function sample_obstacle_position()
    local v = 0.02
    local obstacle_dynamic_collection = simGetCollectionHandle('obstacle_dynamic')
    local obstacle_dynamic_hds = simGetCollectionObjects(obstacle_dynamic_collection)
    for i=1, #obstacle_dynamic_hds, 1 do
        obs_pos = simGetObjectPosition(obstacle_dynamic_hds[i], -1)

        local x = math.abs(obs_pos[1])
        local y = math.abs(obs_pos[2])

        local bound = 1.5
        if x < 2.5 and y < 2.5 then 
            obs_pos[1] = (math.random()-0.5)*2 * bound --+ obs_pos[1]
            obs_pos[2] = (math.random()-0.5)*2 * bound--+ obs_pos[2]

            if obs_pos[1] > bound then
                obs_pos[1] = bound
            elseif obs_pos[1] < -bound then 
                obs_pos[1] = -bound
            end

            if obs_pos[2] > bound then
                obs_pos[2] = bound
            elseif obs_pos[2] < -bound then 
                obs_pos[2] = -bound
            end
        end
        -- print(obs_pos[1], obs_pos[2])
        simSetObjectPosition(obstacle_dynamic_hds[i], -1, obs_pos)
    end
end

function generate_path()
    init_params(2, 8, 'centauro', 'obstacle_all', true)
    task_hd, state_dim = init_task('centauro','task_1')
    path = compute_path(task_hd, 10)
    print ('path found ', #path)
    -- displayInfo('finish 1 '..#path)

    for i=1, 30, 1 do 
        applyPath(task_hd, path, 0.1)
    end
    simExtOMPL_destroyTask(task_hd)

    return path
end

function applyPath(task_hd, path, speed)
    -- simSetModelProperty(robot_hd, 32)

    local state = {}
    for i=1,#path-state_dim,state_dim do
        for j=1,state_dim,1 do
            state[j]=path[i+j-1]
        end
        do_action_hl(_robot_hd, state)
        -- res = simExtOMPL_writeState(task_hd, state) 
        -- pos = {}
        -- pos[1] = state[1]
        -- pos[2] = state[2]
        -- pos[3] = 0
        -- print (pos[1])
        -- simSetObjectPosition(robot_hd, -1, pos)
        -- sleep (0.005)
        sleep(speed)
        simSwitchThread()
    end
    -- simSetModelProperty(robot_hd, 0)
end

function start()
    -- sleep (3)
    -- print('reset')
    _fake_robot_hd = simGetObjectHandle('fake_robot')
    _robot_hd = simGetObjectHandle('centauro')
    _robot_body_hd = simGetObjectHandle('body_ref')
    _target_hd = simGetObjectHandle('target')
    _joint_hds = get_joint_hds(16)

    _start_pos = simGetObjectPosition(_robot_hd, -1)
    _start_ori = simGetObjectQuaternion(_robot_hd,-1)
    _start_joint_values = get_joint_values(_joint_hds)

    _collection_hd = simGetCollectionHandle('obstacle_all')
    _obstacles_hds = simGetCollectionObjects(_collection_hd)
    -- print (_start_pos[1], _start_pos[2])
end

function sample_initial_poses(radius)

    sample_obstacle_position()

    local robot_pos = {}
    robot_pos[1] = (math.random() - 0.5) * 1
    robot_pos[2] = (math.random() - 0.5) * 1
    robot_pos[3] = _start_pos[3]

    local robot_ori = {}
    robot_ori[1] = _start_ori[1]
    robot_ori[2] = _start_ori[2]
    robot_ori[3] = (math.random() - 0.5) *2 * math.pi    --_start_ori[3]
    robot_ori[4] = _start_ori[4]

    simSetObjectPosition(_robot_hd, -1, robot_pos)
    simSetObjectQuaternion(_robot_hd, -1, robot_ori)
    -- set_joint_values(_joint_hds, _start_joint_values)

    local target_pos = {}
    target_pos[1] = (math.random() - 0.5) * radius + robot_pos[1]
    target_pos[2] = (math.random() - 0.5) * radius + robot_pos[2]
    target_pos[3] = _start_pos[3]

    local target_ori = {} 
    target_ori[1] = _start_ori[1] 
    target_ori[2] = _start_ori[2]
    target_ori[3] = robot_ori[3] --(math.random() - 0.5) * 2 * math.pi
    target_ori[4] = _start_ori[4]

    simSetObjectPosition(_target_hd,-1,target_pos)
    simSetObjectQuaternion(_target_hd, -1, target_ori)
    simSetObjectPosition(_fake_robot_hd,-1,target_pos)
    simSetObjectQuaternion(_fake_robot_hd, -1, target_ori)

    local res_robot = simCheckCollision(_robot_hd, _collection_hd)
    local res_target = simCheckCollision(_fake_robot_hd, _collection_hd)

    -- print (res_robot, res_target)
    return res_robot+res_target

end

function init(radius)
    local init_value = 1
    while (init_value ~= 0) do
        init_value = sample_initial_poses(radius)
    end

    -- print('reset')
    -- print(_start_pos[1], _start_pos[2], _start_pos[3])
    -- print(_start_joint_values[9], _start_joint_values[10], _start_joint_values[11], _start_joint_values[12])

    -- sleep(5)
    -- simSetModelProperty(_robot_hd, 32)
    -- g_path = generate_path()
end

initialized = false

-- get_obstacle_info(nil, nil, nil, nil)
start()
init(1.5)
-- simSetModelProperty(_robot_hd, 32)

-- print(_start_pos[1], _start_pos[2], _start_ori[3])
-- for i=1, 10, 1 do
-- --     -- i = 0
--     action = {_start_pos[1], _start_pos[2], 0, 0, 0.1}
--     do_action_hl(_robot_hd, action)
--     simSwitchThread()
--     sleep(3)
-- end
-- start()

-- init()

-- for i=1, 5000, 1 do
--     action = {0,0,0,0,0,0,0,0}
--     for j = 1, 8, 1 do
--         action[j] = (math.random()-0.5)*2
--     end
--     -- action = {-1,1,-1,1,0,0,0,0}
--     print (do_action(robot_hd, action))
-- end


while simGetSimulationState()~=sim_simulation_advancing_abouttostop do
    -- do something in here
    simSwitchThread()
end




