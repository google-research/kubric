URDF_TEMPLATE = """
<robot name="{id}">
    <link name="base">
        <contact>
            <lateral_friction value="{friction}" />  
        </contact>
        <inertial>
            <origin xyz="{center_mass[0]} {center_mass[1]} {center_mass[2]}" />
            <mass value="{mass}" />
            <inertia ixx="{inertia[0][0]}" ixy="{inertia[0][1]}" 
                     ixz="{inertia[0][2]}" iyy="{inertia[1][1]}" 
                     iyz="{inertia[1][2]}" izz="{inertia[2][2]}" />
        </inertial>
        <visual>
            <origin xyz="{center_mass[0]} {center_mass[1]} {center_mass[2]}" />
            <geometry>
                <mesh filename="collision_geometry.obj" />
            </geometry>
        </visual>
        <collision>
            <origin xyz="{center_mass[0]} {center_mass[1]} {center_mass[2]}" />
            <geometry>
                <mesh filename="collision_geometry.obj" />
            </geometry>
        </collision>
    </link>
</robot>
"""