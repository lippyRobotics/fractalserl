import wandb
import pandas as pd
import sys


def upload_runs_from_csv(filename, project_name):
    df = pd.read_csv(filename)

    # Group by Name
    grouped = df.groupby("Name")

    for run_name, group in grouped:
        # Optional: sort if order matters (e.g., by step or time)
        group = group.reset_index(drop=True)

        with wandb.init(
            project=project_name,
            name=str(run_name),
            reinit=True
        ) as run:

            for step_idx, (_, row) in enumerate(group.iterrows()):
                row_dict = row.to_dict()

                log_data = {
                    "success_rate": row_dict.get("success_rate"),
                    "Steps": row_dict.get("Steps"),
                    "Time (min)": row_dict.get("Time (min)"),
                    "Time (sec)": row_dict.get("Time (sec)")
                }

                # Log each row as a step in the same run
                wandb.log(log_data, step=step_idx)



def main():
    if len(sys.argv) != 3:
        print("Usage: python upload_eval_from_csv.py <csv_path> <wandb_project>")
        return

    filename = sys.argv[1]
    project_name = sys.argv[2]


    upload_runs_from_csv(filename, project_name)

if __name__ == "__main__":
    main()