from imageio import readpgm
from termshow import show, ANSI_COLOURS

if __name__ == "__main__":
    import sys

    if len(sys.args) <= 1:
        print(f"Usage {sys.args[0]} <filename>")
        sys.exit(1)

    show(readpgm(sys.args[1]), ANSICOLOURS)