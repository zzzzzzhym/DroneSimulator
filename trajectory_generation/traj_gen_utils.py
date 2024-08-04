import os

def get_dir_from_traj_gen(path_segments: list[str]) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    target_path = os.path.dirname(current_dir)  # src code root dir
    for seg in path_segments:
        target_path = os.path.join((target_path), seg)
    return target_path

if __name__ == "__main__":
    print(get_dir_from_traj_gen([]))
