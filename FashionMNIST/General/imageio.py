# Write PGM images, with the classification as the comment.
# d - the 2d list to write.
# filename - the file to write to.
# classification - the classification of the image
def writepgm(d, filename, classification):
    buffer = 2147483647 # 2 megabytes
    white = 255
    x = len(d)
    y = len(d[0])

# Open our file.
    f = open(filename, mode='w', buffering=buffer)

# Write header.
    f.write('P2\n')
    f.write(f'# {classification}\n')
    f.write(f'{x} {y}\n')
    f.write(f'{white}\n')

# Write out 2d list.
    for j in range(y):
        for i in range(x):
            v = d[j][i]
            quantised = int(v * (white))
            
            # Protect against having colour values outside our range.
            if quantised < 0:
                quantised = 0
            if quantised >= white:
                quantised = white

            f.write(f'{quantised} ')
        f.write('\n')

# Tidy up.
    f.close()