'''
This script convert absolute paths to relative in recorded csv files.
'''
import sys
import os


def usage():
    '''
    print usage
    '''
    print("This script reads csv data from standard input and write to standard output")
    print("\tUsage: cat INPUTFILE | {} > OUTPUTFILE".format(sys.argv[0]))


def make_relative(path):
    '''
    make absolute path into relative path
    '''
    if not os.path.isabs(path):
        return path
    img_path, img_file = os.path.split(path)
    img_path, img_dir = os.path.split(img_path)
    return os.path.join(img_dir, img_file)


def main():
    '''
    main function
    '''
    if len(sys.argv) > 1 and (sys.argv[1] == '-h' or sys.argv[1] == '--help'):
        usage()
        return
    for row in sys.stdin:
        cols = row.split(',')
        if len(cols) != 7:
            continue
        for i in range(3):
            cols[i] = make_relative(cols[i])
        sys.stdout.write(','.join(cols))


if __name__ == '__main__':
    main()
