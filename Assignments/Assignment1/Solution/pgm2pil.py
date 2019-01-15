import PIL.Image
import numpy

def pgm2pil(filename):

    try:
        inFile = open(filename)

        header = None
        size = None
        maxGray = None
        data = []

        for line in inFile:
            stripped = line.strip()

            if stripped[0] == '#': 
                continue
            elif (header == None) or (size == None) or (maxGray == None): 
                header_elem = stripped.split()
                if header == None:
                    if header_elem[0] != 'P2': return None
                    header = header_elem[0]
                    if len(header_elem) > 1: 
                        header_elem = header_elem[1:]
                    else:
                        continue
                    
                if size == None:
                    size = list(map(int, header_elem[0:2]))
                    if len(header_elem) > 2: 
                        header_elem = header_elem[2:]
                    else:
                        continue

                if maxGray == None:
                    maxGray = int(header_elem[0])
                            
            else:
                for item in stripped.split():
                    data.append(int(item.strip()))

        # original had flipped upside-down which seems incorrect
        data = numpy.reshape(data, (size[1],size[0]))/float(maxGray)*255
        # return numpy.flipud(data)
        return data

        
    except:
        #print('Shit')
        pass

    return None

def imageOpenWrapper(fname):
    pgm = pgm2pil(fname)
    if pgm is not None:
        return PIL.Image.fromarray(pgm)
    return origImageOpen(fname)

origImageOpen = PIL.Image.open
PIL.Image.open = imageOpenWrapper
