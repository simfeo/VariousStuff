# ArmASCII8 to utf converter
# specify input and output documents
# tested on python 3.7
# converted map logic taken from http://unicodenow.com/
import sys

arm_map = {}

for i in range(178,253,2):
    arm_map[i] = int(1328 + (i-176)/2)
    arm_map[i+1] = int(1376 + (i-176)/2)

def main(infile = None, outfile = None):
    if not infile:
        infile = sys.argv[1]
    
    if not outfile:
        outfile = sys.argv[2]
        
    f = open(infile,'r', encoding="utf-8")
    lines = f.readlines()
    f.close()

    f = open(outfile, 'w', encoding='utf-8')
    for line in lines:
        for l in line:
            if ord(l) in arm_map:
                f.write(chr(arm_map[ord(l)]))
            else:
                f.write(l)

    f.close()
    print ("job is done!")
    
if __name__ == "__main__":
    main()
