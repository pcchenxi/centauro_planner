package.path=package.path .. ";/home/xi/workspace/a3c_palnner/environment/?.lua"
require("get_set")

-- action: vx, vy, vw, vl

step = 0.05
dx = step
dy = step
dh = step 
dl = step
dw = math.pi/180 * 1
da = math.pi/180 * 10

collision_hd_1 = simGetCollectionHandle('centauro')
collision_hd_2 = simGetCollectionHandle('obstacle_all')

control_joint_hds = get_joint_hds(8)
control_joint_hd_all = get_joint_hds(12)

function is_valid()
    local valid = true
    local reasom = ''
    local res=simCheckCollision(collision_hd_1, collision_hd_2)
    reasom = 'pass'
    if res > 0 then 
        valid = false
        reasom = 'collision'
    end

    for i=1, #control_joint_hd_all, 1 do
        local joint_position = simGetObjectPosition(control_joint_hd_all[i], -1)
        if joint_position[3] < 0.08 then
            valid = false
            reasom = 'body down '..i
            break
        end
    end
    -- print(reasom, valid)
    return valid
end

function action_rl(robot_hd, current_pos, current_ori, action)
    local sample_pose = {}
    sample_pose[1] = dx*action[1]
    sample_pose[2] = dy*action[2] 
    sample_pose[3] = dh*action[4] 

    local sample_ori = {}
    sample_ori[1] = current_ori[1] 
    sample_ori[2] = current_ori[2]   
    sample_ori[3] = current_ori[3] + dw*action[3] 
    sample_ori[4] = current_ori[4]

    simSetObjectPosition(robot_hd,robot_hd,sample_pose)
    simSetObjectQuaternion(robot_hd,-1,sample_ori)

    -- return sample_pose, sample_ori
end

function action_planner(robot_hd, current_pos, current_ori, action)
    -- print(#action)           
    -- print (action[1], action[2], action[3], action[4], action[5])
    local sample_pose = {}
    sample_pose[1] = action[1]
    sample_pose[2] = action[2] 
    sample_pose[3] = action[4] 

    local sample_ori = {}
    sample_ori[1] = current_ori[1] 
    sample_ori[2] = current_ori[2]   
    sample_ori[3] = action[3] 
    sample_ori[4] = current_ori[4]

    simSetObjectPosition(robot_hd, -1, sample_pose)
    simSetObjectQuaternion(robot_hd, -1, sample_ori)
end


function do_action_hl(robot_hd, action)
    local current_pos=simGetObjectPosition(robot_hd,-1)
    local current_ori=simGetObjectQuaternion(robot_hd,-1)
    -- local current_joints = get_joint_positions(joint_hds)
    current_joints = start_joints
    local h, l = get_current_hl(robot_hd, joint_hds)

    action_planner(robot_hd, current_pos, current_ori, action)

    local tilt_pos = {}
    local foot_pos = {}

    -- sample feet and joint --
    local r0 = 0.15 --math.sqrt(knee_pose[1]^2 + knee_pose[2]^2)
    local r1 = 0.1 --math.sqrt(ankle_pose[1]^2 + ankle_pose[2]^2)    

    -- for i=1, 4, 1 do
    local dummy_ankle_hd = simGetObjectHandle('temp_ankle')
    local dummy_wheel_hd = simGetObjectHandle('temp_wheel')
    local ankle_hd = simGetObjectHandle('ankle_pitch_1')
    local hip_hd = simGetObjectHandle('hip_pitch_1')
    local knee_hd = simGetObjectHandle('knee_pitch_1')

    hip_pose = simGetObjectPosition(hip_hd, robot_hd)

    -- set dummy for target ankle position
    local ankle_target = {}
    ankle_target[1] = hip_pose[1] + action[5]
    ankle_target[2] = hip_pose[2]
    ankle_target[3] = 0.17 - action[4]
    simSetObjectPosition(dummy_ankle_hd, robot_hd, ankle_target)
    -- print('ankle_target:', i, ankle_target[1], ankle_target[2], ankle_target[3])

    -- set dummy for target wheel position
    local wheel_target = ankle_target
    wheel_target[3] = wheel_target[3] - 0.1175
    simSetObjectPosition(dummy_wheel_hd, robot_hd, wheel_target)

    local ankle_in_hip = simGetObjectPosition(dummy_ankle_hd, hip_hd)
    -- print('ankle_target:', i, ankle_in_hip[1], ankle_in_hip[2], ankle_in_hip[3])

    local knee_x, knee_y = get_intersection_point(0, 0, ankle_in_hip[1], ankle_in_hip[2], r0, r1)
    -- print('knee_target:', i, knee_x, knee_y)

    if knee_x == -1 or knee_x ~= knee_x then
        return false
    end

    local angle_thigh = 0
    angle_thigh = -math.atan2(knee_x, knee_y)
    simSetJointPosition(hip_hd, angle_thigh)

    local ankle_in_knee = simGetObjectPosition(dummy_ankle_hd, knee_hd)
    local angle_knee = math.atan2(ankle_in_knee[2], ankle_in_knee[1])
    simSetJointPosition(knee_hd, angle_knee)

    local wheel_in_ankle = simGetObjectPosition(dummy_wheel_hd, ankle_hd)
    local angle_ankle = math.atan2(wheel_in_ankle[2], wheel_in_ankle[1])      
    simSetJointPosition(ankle_hd, angle_ankle)


    local ankle2_hd = simGetObjectHandle('ankle_pitch_2')
    local hip2_hd = simGetObjectHandle('hip_pitch_2')
    local knee2_hd = simGetObjectHandle('knee_pitch_2')
    simSetJointPosition(hip2_hd, -angle_thigh)
    simSetJointPosition(knee2_hd, -angle_knee)
    simSetJointPosition(ankle2_hd, -angle_ankle)

    local ankle3_hd = simGetObjectHandle('ankle_pitch_3')
    local hip3_hd = simGetObjectHandle('hip_pitch_3')
    local knee3_hd = simGetObjectHandle('knee_pitch_3')
    simSetJointPosition(hip3_hd, -angle_thigh)
    simSetJointPosition(knee3_hd, -angle_knee)
    simSetJointPosition(ankle3_hd, -angle_ankle)

    local ankle4_hd = simGetObjectHandle('ankle_pitch_4')
    local hip4_hd = simGetObjectHandle('hip_pitch_4')
    local knee4_hd = simGetObjectHandle('knee_pitch_4')
    simSetJointPosition(hip4_hd, angle_thigh)
    simSetJointPosition(knee4_hd, angle_knee)
    simSetJointPosition(ankle4_hd, angle_ankle)

    -- check collision --
    if is_valid() then 
        return true
    else
        -- displayInfo('collide '..i..' '..foot_pos[1]..' '..foot_pos[2] )
        return false      
    end
end

get_intersection_point=function(x0, y0, x1, y1, r0, r1)
    local d=math.sqrt((x1-x0)^2 + (y1-y0)^2)
    if d>(r0+r1) then
        return -1, -1
    end
    
    local a=(r0^2-r1^2+d^2)/(2*d)
    local h=math.sqrt(r0^2-a^2)
    local x2=x0+a*(x1-x0)/d   
    local y2=y0+a*(y1-y0)/d   
    local x3=x2+h*(y1-y0)/d       -- also x3=x2-h*(y1-y0)/d
    local y3=y2-h*(x1-x0)/d       -- also y3=y2+h*(x1-x0)/d

    if math.abs(x3)>math.abs(x2) then
        return x3, y3
    else 
        return x2, y2 
    end
end

function get_current_hl(robot_hd, joint_hds)
    local ankle_hd = simGetObjectHandle('ankle_pitch_1')
    local hip_hd = simGetObjectHandle('hip_pitch_1')
    local foot_pos = simGetObjectPosition(ankle_hd, hip_hd)

    local current_pos=simGetObjectPosition(robot_hd,-1)

    return current_pos[3], math.abs(foot_pos[1])
end
