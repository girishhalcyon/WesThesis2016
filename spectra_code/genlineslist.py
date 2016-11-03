import numpy as np


def readlines(fname, titlecols = [3,4]):
    temptitles = np.loadtxt(fname, usecols = titlecols, dtype = 'string')
    clambdas = np.loadtxt(fname, usecols = [0])
    ions = [temptitles[i,0] + temptitles[i,1] for i in range(0,len(clambdas))]
    return ions, clambdas

def findmatch(goodions, goodlines, reffile, writename = 'plotfe1linelist.txt'):
    searchrange = len(goodlines)
    refions, reflambdas = readlines(reffile)
    reflambdas = 10.0*reflambdas
    matchindices = [np.where(reflambdas == goodlines[i])[0] for i in range(0, searchrange)]
    print matchindices
    j = 0
    with open(reffile, 'r') as readfile, open(writename, 'a') as writefile:
        for line in readfile:
            if j in matchindices:
                writefile.write(line)
            j += 1

    print 'Wrote lines to ' + writename

if __name__ == '__main__':
    goodions, goodlines = readlines('fe1goodlines.txt', titlecols = [1,2])
    findmatch(goodions, goodlines, 'FeIlines.txt')
