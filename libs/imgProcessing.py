import math

class imgProcessing:
    @staticmethod
    def getCropBox(width, height, nsplit):
        if math.sqrt(nsplit) % 1 == 0:
            ncols = int(math.sqrt(nsplit))
            nrows = ncols
        elif nsplit % 2 == 0:
            ncols = int(nsplit / 2)
            nrows = int(nsplit / ncols)
        else:
            ncols = nsplit
            nrows = 1

        crop_width = int(width / ncols)
        crop_height = int(height / nrows)
        
        boxes = []
        for i in range(nrows):
            for j in range(ncols):
                if i == nrows-1 and j == ncols-1:
                    box = [j*crop_width, i*crop_height, width, height]
                elif j == ncols-1:
                    box = [j*crop_width, i*crop_height, width, (i+1)*crop_height]
                elif i == nrows-1:
                    box = [j*crop_width, i*crop_height, (j+1)*crop_width, height]
                else:
                    box = [j*crop_width, i*crop_height, (j+1)*crop_width, (i+1)*crop_height]
                boxes.append(box)
                
        return boxes