import os


# TODO: (Malcolm 2023-08-11) add voices
def path_formatter(path, i, transpose):
    # TODO: (Malcolm 2023-08-11) set i or remove
    return (
        os.path.splitext(os.path.basename(path))[0]
        + f"_transpose={transpose}_{i+1:03d}"
    )
