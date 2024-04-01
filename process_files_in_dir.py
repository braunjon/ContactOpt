import argparse
import os
import subprocess
import sys

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--in_dir", type=str)
    args = argparser.parse_args()

    in_dir = args.in_dir

    non_cleaned_files = os.listdir(in_dir)

    for f in non_cleaned_files:
        sequence_non_cleaned_fp = os.path.join(in_dir, f)
        # call contactopt to optimize pose
        contactopt_script_fp = "process_pose.sh"
        args_run_py = ['sh', contactopt_script_fp]
        args_run_py += ['--in_pose_fp', sequence_non_cleaned_fp]

        p = subprocess.Popen(args_run_py, stdout=sys.stdout, stderr=sys.stderr)
        p.wait()
        out, err = p.communicate()
