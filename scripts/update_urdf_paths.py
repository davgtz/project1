import re
import os

def update_mesh_paths(input_file, output_file, project_path):
    
    # Regular expression to capture the portion before /robot/ and the part starting from /robot/
    pattern = re.compile(r'filename="(?:[^/]*?/)?(robot/[^"]+)"')

    with open(input_file, 'r') as file:
        content = file.read()

    def replace_path(match):

        new_path = os.path.join(project_path, match.group(1))
        return f'filename="{new_path}"'

    # Apply the regex substitution
    updated_content = pattern.sub(replace_path, content)

    with open(output_file, 'w') as file:
        file.write(updated_content)

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_file = "robot/urdfs/ur5_w.urdf"
    output_file = "robot/urdfs/ur5.urdf"
    
    project_path = os.getcwd()
    print(project_path)

    update_mesh_paths(input_file, output_file, project_path)
    print(f"Updated URDF file saved to {output_file}")

    input_file = "robot/urdfs/kuka_lwr_w.urdf"
    output_file = "robot/urdfs/kuka_lwr.urdf"

    update_mesh_paths(input_file, output_file, project_path)
    print(f"Updated URDF file saved to {output_file}")