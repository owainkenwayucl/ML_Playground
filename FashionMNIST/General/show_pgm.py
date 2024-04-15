from imageio import readpgm
from termshow import show, ANSI_COLOURS

if __name__ == "__main__":
    import sys

    if len(sys.argv) <= 1:
        print(f"Usage {sys.argv[0]} <filename>")
        sys.exit(1)

    show(readpgm(sys.argv[1]), ANSI_COLOURS)