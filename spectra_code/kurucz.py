import numpy as np

def readion(ioncode, ionname, ionstage, writename, fname = 'gfall18feb16.dat'):
    master = np.loadtxt(fname, dtype = str, usecols = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))
    col0 = master[:,0]
    col1 = master[:,1]
    col2 = master[:,2]
    col3 = master[:,3]
    col4 = master[:,4]
    col5 = master[:,5]
    col6 = master[:,6]
    col7 = master[:,7]
    col8 = master[:,8]
    col9 = master[:,9]
    col10 = master[:,10]
    col11 = master[:,11]
    col12 = master[:,12]
    col13 = master[:,13]

    with open(writename, 'a') as writefile:
        for i in range(0,len(col0)):
            try:
                if col1[i] == (ioncode):
                    elcode = col1[i]
                    minuspos = np.where(col0[i] == '-')
                    lamda = col0[i][:minuspos]
                    loggf = col0[i][minuspos:]
                    elow = col2[i]
                    jlow = col3[i]
                    ehigh = col6[i]
                    jhigh = col7[i]
                    refcode = col12[i][5:]
                if col2[i] == (ioncode):
                    elcode = col2[i]
                    lamda = col0[i]
                    loggf = col1[i]
                    elow = col3[i]
                    jlow = col4[i]
                    ehigh = col7[i]
                    jhigh = col8[i]
                    refcode = col13[i][5:]
                if col3[i] == (ioncode):
                    elcode = col3[i]
                    lamda = col0[i]
                    loggf = col1[i] + '.' + col3[i]
                    elow = col4[i]
                    jlow = col5[i]
                    ehigh = col8[i]
                    jhigh = col9[i]
                    refcode = 'IDK'
                if float(lamda) > 300.0:
                    if float(lamda) < 25000.0:
                        if elcode == ioncode:
                            halfstring1 = '   ' + str(lamda) +  '   ' + str(loggf) +  '  ' + str(elcode) + ' ' + str(ionname) + ' ' + str(ionstage)
                            halfstring2 = '       ' + str(elow) + '   ' + str(jlow) + '    ' + str(ehigh) + '   ' + str(jhigh) + ' ' + str(refcode)
                            fullstring = halfstring1 + halfstring2 + '  \n'
                            writefile.write(fullstring)
                            #print 'Wrote line ' + lamda
                elcode = 0.0
            except:
                pass
#                print 'Failed'
    print 'Wrote to ' + writename

if __name__ == '__main__':
    '''
    readion('1.00', 'H', 'I', 'H_I_lines.txt')

    readion('2.00', 'He', 'I', 'He_I_lines.txt')
    readion('2.01', 'He', 'II', 'He_II_lines.txt')

    readion('6.00', 'C', 'I', 'C_I_lines.txt')
    readion('6.01', 'C', 'II', 'C_II_lines.txt')
    readion('6.02', 'C', 'III', 'C_III_lines.txt')
    readion('6.03', 'C', 'IV', 'C_IV_lines.txt')

    readion('7.00', 'N', 'I', 'N_I_lines.txt')
    readion('7.01', 'N', 'II', 'N_II_lines.txt')

    readion('8.00', 'O', 'I', 'O_I_lines.txt')
    readion('8.01', 'O', 'II', 'O_II_lines.txt')

    readion('12.00', 'Mg', 'I', 'Mg_I_lines.txt')
    readion('12.01', 'Mg', 'II', 'Mg_II_lines.txt')
    readion('12.02', 'Mg', 'III', 'Mg_III_lines.txt')

    readion('13.00', 'Al', 'I', 'Al_I_lines.txt')
    readion('13.01', 'Al', 'II', 'Al_II_lines.txt')
    readion('13.02', 'Al', 'III', 'Al_III_lines.txt')
    readion('13.03', 'Al', 'IV', 'Al_IV_lines.txt')

    readion('14.00', 'Si', 'I', 'Si_I_lines.txt')
    readion('14.01', 'Si', 'II', 'Si_II_lines.txt')
    readion('14.02', 'Si', 'III', 'Si_III_lines.txt')
    readion('14.03', 'Si', 'IV', 'Si_IV_lines.txt')
    readion('14.04', 'Si', 'V', 'Si_V_lines.txt')

    readion('15.00', 'P', 'I', 'P_I_lines.txt')
    readion('15.01', 'P', 'II', 'P_II_lines.txt')
    readion('15.02', 'P', 'III', 'P_III_lines.txt')
    readion('15.03', 'P', 'IV', 'P_IV_lines.txt')
    readion('15.04', 'P', 'V', 'P_V_lines.txt')

    readion('16.00', 'S', 'I', 'S_I_lines.txt')
    readion('16.01', 'S', 'II', 'S_II_lines.txt')
    readion('16.02', 'S', 'III', 'S_III_lines.txt')
    readion('16.03', 'S', 'IV', 'S_IV_lines.txt')
    readion('16.04', 'S', 'V', 'S_V_lines.txt')

    readion('20.00', 'Ca', 'I', 'Ca_I_lines.txt')
    readion('20.01', 'Ca', 'II', 'Ca_II_lines.txt')
    readion('20.02', 'Ca', 'III', 'Ca_III_lines.txt')

    readion('21.00', 'Sc', 'I', 'Sc_I_lines.txt')
    readion('21.01', 'Sc', 'II', 'Sc_II_lines.txt')
    readion('21.02', 'Sc', 'III', 'Sc_III_lines.txt')

    readion('22.00', 'Ti', 'I', 'Ti_I_lines.txt')
    readion('22.01', 'Ti', 'II', 'Ti_II_lines.txt')
    readion('22.02', 'Ti', 'III', 'Ti_III_lines.txt')
    readion('22.03', 'Ti', 'IV', 'Ti_IV_lines.txt')

    readion('23.00', 'V', 'I', 'V_I_lines.txt')
    readion('23.01', 'V', 'II', 'V_II_lines.txt')
    readion('23.02', 'V', 'III', 'V_III_lines.txt')
    readion('23.03', 'V', 'IV', 'V_IV_lines.txt')

    readion('24.00', 'Cr', 'I', 'Cr_I_lines.txt')
    readion('24.01', 'Cr', 'II', 'Cr_II_lines.txt')
    readion('24.02', 'Cr', 'III', 'Cr_III_lines.txt')
    readion('24.03', 'Cr', 'IV', 'Cr_IV_lines.txt')

    readion('25.00', 'Mn', 'I', 'Mn_I_lines.txt')
    readion('25.01', 'Mn', 'II', 'Mn_II_lines.txt')
    readion('25.02', 'Mn', 'III', 'Mn_III_lines.txt')
    readion('25.03', 'Mn', 'IV', 'Mn_IV_lines.txt')

    readion('26.00', 'Fe', 'I', 'Fe_I_lines.txt')
    readion('26.01', 'Fe', 'II', 'Fe_II_lines.txt')
    readion('26.02', 'Fe', 'III', 'Fe_III_lines.txt')
    readion('26.03', 'Fe', 'IV', 'Fe_IV_lines.txt')
    '''
    readion('28.00', 'Ni', 'I', 'Ni_I_lines.txt')
    readion('28.01', 'Ni', 'II', 'Ni_II_lines.txt')
    readion('28.02', 'Ni', 'III', 'Ni_III_lines.txt')
