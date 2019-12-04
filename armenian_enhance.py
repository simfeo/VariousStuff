arm_map = {}

for i in range(178,253,2):
    arm_map[i] = int(1328 + (i-176)/2)
    arm_map[i+1] = int(1376 + (i-176)/2)

f = open('C:\media\document.xml','r', encoding="utf-8")
lines = f.readlines()
f.close()


f = open("out.xml", 'w', encoding='utf-8')
for line in lines:
    for l in line:
        if ord(l) in arm_map:
            f.write(chr(arm_map[ord(l)]))
        else:
            f.write(l)

f.close()