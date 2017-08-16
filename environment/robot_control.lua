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


function do_action_hl(robot_hd, action)
    local current_pos=simGetObjectPosition(robot_hd,-1)
    local current_ori=simGetObjectQuaternion(robot_hd,-1)
    -- local current_joints = get_joint_positions(joint_hds)
    current_joints = start_joints
    local h, l = get_current_hl(robot_hd, joint_hds)

    local sample_pose = {}
    sample_pose[1] = dx*action[1]
    sample_pose[2] = dy*action[2] 
    sample_pose[3] = dh*action[4] 

    local sample_ori = {}
    sample_ori[1] = current_ori[1] 
    sample_ori[2] = current_ori[2]   
    sample_ori[3] = current_ori[3] + dw*action[3] 
    sample_ori[4] = current_ori[4]

    -- print (sample_ori[3])
    simSetObjectPosition(robot_hd,robot_hd,sample_pose)
    simSetObjectQuaternion(robot_hd,-1,sample_ori)

    local r0 = 0.07238 --math.sqrt(knee_pos[1]^2 + knee_pos[2]^2)
    local r1 = 0.10545 --math.sqrt(tip_pos[1]^2 + tip_pos[2]^2)

    local tilt_pos = {}
    local foot_pos = {}

    -- sample feet and joint --
    for i=1, 4, 1 do
        local leg_joints=get_leg_hds(i) -- pan, tilt, knee, ankle

        tilt_pos = simGetObjectPosition(leg_joints[2], -1)
        foot_pos = simGetObjectPosition(leg_joints[4], leg_joints[2])

        foot_pos[1] = foot_pos[1] + dl*action[5] 
        foot_pos[2] = tilt_pos[3] - 0.0444

        if foot_pos[2] < 0 then
            displayInfo('too low '..i..' '..foot_pos[1]..' '..foot_pos[2] )
            restore_pose(robot_hd, joint_hds, current_pos, current_ori, current_joints)
            return 'f'
        end

        local knee_x, knee_y = get_intersection_point(0, 0, foot_pos[1], foot_pos[2], r0, r1)

        if knee_x == -1 or knee_x ~= knee_x then
            displayInfo('no pose found '..i..' '..foot_pos[1]..' '..foot_pos[2] )
            restore_pose(robot_hd, joint_hds, current_pos, current_ori, current_joints)
            return 'f'
        end

        local foot_x_fromknee = foot_pos[1]-knee_x
        local foot_y_fromknee = foot_pos[2]-knee_y

        -- displayInfo('knee pos: '..foot_x_fromknee..' '..foot_y_fromknee)

        local angle_thigh = math.atan(knee_y/knee_x)
        local angle_knee = math.atan(foot_y_fromknee/foot_x_fromknee)
        if angle_knee<0 then 
            angle_knee = angle_knee + math.pi 
        end

        simSetJointPosition(leg_joints[2], angle_thigh)
        simSetJointPosition(leg_joints[3], angle_knee-angle_thigh)

        -- print (i, angle_thigh, angle_knee-angle_thigh)
        -- local real_foot_pos = 
    end

    -- check collision --
    if is_valid() then 
        return 't'
    else
        restore_pose(robot_hd, joint_hds, current_pos, current_ori, current_joints)
        -- displayInfo('collide '..i..' '..foot_pos[1]..' '..foot_pos[2] )
        return 'f'      
    end
end



function do_action(robot_hd, action)
    local current_pos=simGetObjectPosition(robot_hd,-1)
    local current_ori=simGetObjectQuaternion(robot_hd,-1)
    local current_joints = get_joint_values(control_joint_hds)

    -- -- robot position
    -- local sample_pose = {}
    -- sample_pose[1] = dx*action[1]
    -- sample_pose[2] = dy*action[2] 
    -- sample_pose[3] =  current_pos[3] --dh*action[4] 

    -- -- robot orientation
    -- local sample_ori = {}
    -- sample_ori[1] = current_ori[1] 
    -- sample_ori[2] = current_ori[2]   
    -- sample_ori[3] = current_ori[3] + dw*action[3] 
    -- sample_ori[4] = current_ori[4]
    -- simSetObjectPosition(robot_hd, robot_hd, sample_pose)
    -- simSetObjectQuaternion(robot_hd, -1, sample_ori)

    -- joint values
    local joint_values = {}
    for i=1, #control_joint_hds, 1 do 
        joint_values[i] = current_joints[i] + da*action[i]
    end
    set_joint_values(control_joint_hds, joint_values)

    local new_joint_values = get_joint_values(control_joint_hds)
    local new_pos=simGetObjectPosition(robot_hd,-1)

    res = 'f'
    if is_valid() then
        res = 't'
    end

    return res
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

    return x3, y3
end

function get_current_hl(robot_hd, joint_hds)
    local ankle_hd = simGetObjectHandle('ankle_pitch_1')
    local hip_hd = simGetObjectHandle('hip_pitch_1')
    local foot_pos = simGetObjectPosition(ankle_hd, hip_hd)

    local current_pos=simGetObjectPosition(robot_hd,-1)

    return current_pos[3], math.abs(foot_pos[1])
end
