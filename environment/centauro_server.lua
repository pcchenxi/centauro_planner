package.path=package.path .. ";/home/xi/workspace/centauro_planer/environment/?.lua"
require("common_functions")
require("ompl_functions")
require("robot_control")

-- simSetThreadSwitchTiming(2) 
-- simExtRemoteApiStart(19999)
function reset(inInts,inFloats,inStrings,inBuffer)
    local level = inFloats[1]
    init()
    return {}, {}, {}, ''
end

function step(inInts,inFloats,inStrings,inBuffer)
    res = do_action(_robot_hd, inFloats)
    -- sample_obstacle_position(obs_hds, #obs_hds)

    return {}, {}, {}, res
end


-- function get_obstacle_info(inInts,inFloats,inStrings,inBuffer)
--     collection_hd = simGetCollectionHandle('obstacle_all')
--     obstacles_hds = simGetCollectionObjects(obstacle_low_hd)

--     for i=1, #_obstacles_hds, 1 do 
--         local pos = simGetObjectPosition(_obstacles_hds[i])


--     end

--     return {}, {}, {}, res
-- end

function get_robot_state(inInts,inFloats,inStrings,inBuffer)
    local pos =simGetObjectPosition(_robot_body_hd,-1)
    local ori =simGetObjectQuaternion(_robot_body_hd,-1)
    local joint_pose = get_joint_values(_joint_hds)

    local state = {}
    state[1] = pos[1]
    state[2] = pos[2]
    state[3] = pos[3]
    state[4] = ori[3]

    for i=1, #_joint_hds, 1 do
        state[#state+1] = _joint_hds[i]
    end

    -- print ('in get robot state:', #state[3])
    return {}, state, {}, ''
end

function generate_path()
    init_params(2, 8, 'centauro', 'obstacle_all', true)
    task_hd, state_dim = init_task('centauro','task_1')
    path = compute_path(task_hd, 60)
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
    sleep (3)
    print('reset')
    _robot_hd = simGetObjectHandle('centauro')
    _robot_body_hd = simGetObjectHandle('body_ref')
    _target_hd = simGetObjectHandle('target')
    _joint_hds = get_joint_hds(8)

    _start_pos = simGetObjectPosition(_robot_hd, -1)
    _start_ori = simGetObjectQuaternion(_robot_hd,-1)
    _start_joint_values = get_joint_values(_joint_hds)

    _collection_hd = simGetCollectionHandle('obstacle_all')
    _obstacles_hds = simGetCollectionObjects(_collection_hd)
    -- print (_start_pos[1], _start_pos[2])
end

function init()
    if initialized == false then 
        start()
        initialized = true
    end 
    -- -- forbidThreadSwitches(true)
    -- -- set target --
    -- local robot_pos = _start_pos
    -- local robot_ori = _start_ori

    -- robot_pos[1] = _start_pos[1] 
    -- robot_pos[2] = _start_pos[2]
    -- robot_pos[3] = _start_pos[3]

    -- robot_ori[3] = 0

    -- local target_pos = {}
    -- target_pos[1] = (math.random() - 0.5) * 2 + robot_pos[1]
    -- target_pos[2] = (math.random() - 0.5) * 2 + robot_pos[2]
    -- target_pos[3] = _start_pos[3]

    -- simSetObjectPosition(_target_hd,-1,target_pos)
    simSetModelProperty(_robot_hd, 32)
    g_path = generate_path()
end

initialized = false
init()
-- simSetModelProperty(_robot_hd, 32)

-- print(_start_pos[1], _start_pos[2], _start_ori[3])
-- -- for i=1, 10, 1 do
--     -- i = 0
-- action = {_start_pos[1], _start_pos[2], 0, -0.2, 0.25}
-- do_action_hl(_robot_hd, action)
-- simSwitchThread()
-- sleep(3)
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




